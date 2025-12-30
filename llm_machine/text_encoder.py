# vt_siglip/text_encoder.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


DEFAULT_LLM_ID = "meta-llama/Llama-3.2-3B-Instruct"   


class LLMTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_ID,          # ← 기본값 지정
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        train_llm: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = (
            "q_proj",  "v_proj"

        ),
        pooling: str = "mean",
        hf_token: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name

        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            token=token,                      
            trust_remote_code=False,
        )
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",       
            token=token,                      
            trust_remote_code=False,
        )

        loaded_id = getattr(self.lm.config, "_name_or_path", "")
        if DEFAULT_LLM_ID not in loaded_id and self.model_name != DEFAULT_LLM_ID:
            print(f"[LLMTextEncoder] Loaded: {loaded_id} (requested: {self.model_name})")
        else:
            print(f"[LLMTextEncoder] ✔ Using LLaMA-3.2-3B-Instruct: {loaded_id}")
        if getattr(self.lm.config, "model_type", "") != "llama":
            raise RuntimeError(f"Loaded model is not a LLaMA family: {self.lm.config.model_type}")

        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_llm = train_llm

        if self.train_llm and use_lora:
            if not _PEFT_AVAILABLE:
                raise RuntimeError("peft 패키지가 필요함: pip install peft")
            lcfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.lm = get_peft_model(self.lm, lcfg)

        if self.train_llm:
            self.lm.train()
        else:
            for p in self.lm.parameters():
                p.requires_grad = False
            self.lm.eval()
    def _lm_device(self) -> torch.device:
        try:
            return next(self.lm.parameters()).device
        except StopIteration:
            # fallback
            return torch.device(self.device if isinstance(self.device, str) else "cpu")
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lm_dev = self._lm_device()  
        with torch.set_grad_enabled(self.train_llm):
            input_ids_dev = input_ids.to(lm_dev, non_blocking=True)
            attention_mask_dev = attention_mask.to(lm_dev, non_blocking=True)

            out = self.lm(
                input_ids=input_ids_dev,
                attention_mask=attention_mask_dev,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.hidden_states[-1]  # [N, L, H]

            if self.pooling == "mean":
                mask = attention_mask_dev.unsqueeze(-1).to(hidden.dtype)  # [N, L, 1]
                summed = (hidden * mask).sum(dim=1)                        # [N, H]
                denom = mask.sum(dim=1).clamp(min=1e-6)                    # [N, 1]
                emb = summed / denom
            else:  
                eos_id = self.tokenizer.eos_token_id
                ids_list = input_ids_dev.tolist()
                idxs = []
                for seq in ids_list:
                    try:
                        last_idx = len(seq) - 1 - seq[::-1].index(eos_id)
                    except ValueError:
                        last_idx = len(seq) - 1
                    idxs.append(last_idx)
                idxs = torch.tensor(idxs, device=hidden.device, dtype=torch.long)
                emb = hidden[torch.arange(hidden.size(0), device=hidden.device), idxs]
            return emb

    def detect_hidden_size(self) -> int:
        lm_dev = self._lm_device()
        tmp = self.tokenizer(["hello"], padding=True, truncation=True, return_tensors="pt")
        tmp = {k: v.to(lm_dev) for k, v in tmp.items()}
        with torch.set_grad_enabled(self.train_llm):
            out = self.lm(
                input_ids=tmp["input_ids"],
                attention_mask=tmp["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return out.hidden_states[-1].size(-1)
