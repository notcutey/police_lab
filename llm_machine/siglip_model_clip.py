import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Hashable, Dict, Any
from networks.RetrievalNet import Token  
from llm_machine.text_encoder import LLMTextEncoder  

@torch.no_grad()
def compute_class_balanced_weights_from_counts(
    label_pos_counts: torch.Tensor,   
    beta: float = 0.999,             
    w_min: float = 0.25,             
    w_max: float = 4.0,               
    device: Optional[torch.device] = None,
) -> torch.Tensor:
  
    if device is None:
        device = label_pos_counts.device if isinstance(label_pos_counts, torch.Tensor) else torch.device("cpu")

    counts = torch.as_tensor(label_pos_counts, dtype=torch.float32, device=device).clamp(min=1.0)
    num = 1.0 - beta
    den = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32, device=device), counts)
    w = num / torch.clamp(den, min=1e-12)

    w = w * (w.numel() / w.sum().clamp(min=1e-12))

    w = torch.clamp(w, min=w_min, max=w_max)
    return w


class VisionTextSigLIP(nn.Module):
    def __init__(
        self,
        token_model: Token,
        text_encoder: LLMTextEncoder,                      # detect_hidden_size(), encode_text(), tokenizer, device 보유
        vision_dim: int = 1024,                            # Token.forward_test 출력 차원 Dv
        proj_out_dim: int = 1024,                          # 공유 임베딩 차원 D
        temperature_init: float = 0.07,
        llm_hidden_size: Optional[int] = None,
        sync_text_device: bool = True,

        # --- Vision projector 옵션 ---
        use_vision_proj: bool = True,
        vision_proj_type: str = "linear",                  # "linear" | "mlp" --> linear를 통한 안정성 확보
        vision_proj_hidden: Optional[int] = None,        
        use_vision_residual: bool = False,                 
        vision_dropout: float = 0.0,                       

        # --- Asymmetric Loss(멀티라벨) 옵션 ---
        use_asl: bool = True,
        asl_gamma_pos: float = 0.0,                        
        asl_gamma_neg: float = 4.0,                       
        asl_clip: float = 0.05,                            
        asl_eps: float = 1e-8,                             
        
        # --- Class-Balanced Weight 옵션 ---
        use_cbw: bool = True,                              
        class_weights: Optional[torch.Tensor] = None,      
    ):
        super().__init__()
        self.vision_encoder = token_model
        self.text_encoder = text_encoder
        self.tokenizer = getattr(text_encoder, "tokenizer", None)

        if llm_hidden_size is None:
            llm_hidden_size = self.text_encoder.detect_hidden_size()

        # ── Text projector (다단계 축소) ───────────────────────────────
        # 예: 3072 -> 2048 -> 1024
        self.text_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, proj_out_dim),
            nn.LayerNorm(proj_out_dim),
        )

        # ── Vision projector (옵션) ────────────────────────────────
        self.use_vision_proj = use_vision_proj
        self.use_vision_residual = use_vision_residual
        self.vision_dim = vision_dim
        self.proj_out_dim = proj_out_dim

        if use_vision_proj:
            if vision_proj_type not in {"linear", "mlp"}:
                raise ValueError(f"Unknown vision_proj_type: {vision_proj_type}")
            self.vision_proj_type = vision_proj_type

            if vision_proj_type == "linear":
                self.vision_proj = nn.Linear(vision_dim, proj_out_dim)
            else:
                hid = vision_proj_hidden or proj_out_dim
                self.vision_proj = nn.Sequential(
                    nn.LayerNorm(vision_dim),
                    nn.Linear(vision_dim, hid),
                    nn.GELU(),
                    nn.Dropout(p=vision_dropout) if vision_dropout > 0 else nn.Identity(),
                    nn.Linear(hid, proj_out_dim),
                )
            if use_vision_residual:
                if vision_dim != proj_out_dim:
                    self.vision_residual = nn.Linear(vision_dim, proj_out_dim, bias=False)
                else:
                    self.vision_residual = nn.Identity()
        else:
            self.vision_proj = None
            self.vision_residual = None
            self.vision_proj_type = "none"

        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature_init, dtype=torch.float32)))

        if sync_text_device:
            vt_dev = next(self.parameters()).device  # 아직 CPU일 수 있음
            try:
                self.text_encoder.to(vt_dev)
                if hasattr(self.text_encoder, "device"):
                    self.text_encoder.device = vt_dev.type if isinstance(vt_dev, torch.device) else str(vt_dev)
            except Exception:
                pass

        self.use_asl = use_asl
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip
        self.asl_eps = asl_eps
        self.use_cbw = use_cbw and (class_weights is not None)
        if self.use_cbw:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    @torch.no_grad()
    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
            self.use_cbw = False
        else:
            self.class_weights = class_weights.float().to(self.class_weights.device if self.class_weights is not None else next(self.parameters()).device)
            self.use_cbw = True

    @torch.no_grad()
    def _to_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def encode_images(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        v = self.vision_encoder.forward_test(images, **kwargs)  # (B, Dv)
        if self.use_vision_proj:
            if isinstance(self.vision_proj, nn.Sequential):
                first_linear = next(m for m in self.vision_proj if isinstance(m, nn.Linear))
                target_dtype = first_linear.weight.dtype
            else:
                target_dtype = self.vision_proj.weight.dtype
            v = v.to(target_dtype)

            v_in = v
            v = self.vision_proj(v)  
            if self.use_vision_residual:
                if self.vision_residual is not None:
                    v = v + self.vision_residual(v_in)
                else:
                    v = v + v_in
        v = F.normalize(v, dim=-1)
        return v

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        t_raw = self.text_encoder.encode_text(input_ids, attention_mask)   # (M, H_lm)
        t_raw = t_raw.to(next(self.text_proj.parameters()).dtype)
        t = self.text_proj(t_raw)                                          # (M, D)
        t = F.normalize(t, dim=-1)
        return t

    @staticmethod
    def _build_targets_from_labels(
        image_label_lists: Sequence[Sequence[Hashable]],
        text_labels: Sequence[Hashable],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        B = len(image_label_lists)
        M = len(text_labels)
        Y = torch.zeros((B, M), dtype=dtype, device=device)

        label_to_col: Dict[Hashable, int] = {}
        for j, lbl in enumerate(text_labels):
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            label_to_col[lbl] = j

        for i, lab_list in enumerate(image_label_lists):
            if isinstance(lab_list, torch.Tensor):
                lab_list = lab_list.detach().cpu().tolist()
            for lbl in set(lab_list):
                j = label_to_col.get(lbl, None)
                if j is not None:
                    Y[i, j] = 1.0
        return Y

    def _asymmetric_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.asl_eps

        
        x_sigmoid = torch.sigmoid(logits)          # p
        xs_pos = x_sigmoid                         # p
        xs_neg = 1.0 - x_sigmoid                   # 1-p
        if self.asl_clip is not None and self.asl_clip > 0:
            xs_neg = (xs_neg + self.asl_clip).clamp(max=1.0)
        log_pos = torch.log(xs_pos.clamp(min=eps))  # log(p)
        log_neg = torch.log(xs_neg.clamp(min=eps))  # log(1-p_clipped)
        gamma_pos = self.asl_gamma_pos
        gamma_neg = self.asl_gamma_neg
        pos_weight = torch.pow(1.0 - xs_pos, gamma_pos)   # (1-p)^γ+
        neg_weight = torch.pow(1.0 - xs_neg, gamma_neg)   # ≈ p^γ-
        if self.use_cbw and (self.class_weights is not None):
            cw = self.class_weights.to(logits.dtype).to(logits.device).unsqueeze(0).expand_as(targets)
        else:
            cw = None
        if cw is not None:
            loss_pos = - targets * pos_weight * log_pos * cw
        else:
            loss_pos = - targets * pos_weight * log_pos
        loss_neg = - (1.0 - targets) * neg_weight * log_neg
        loss = loss_pos + loss_neg
        return loss.mean()

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        image_label_lists: Optional[Sequence[Sequence[Hashable]]] = None,
        text_labels: Optional[Sequence[Hashable]] = None,
        return_embeddings: bool = False,
        **token_ft_kwargs,
    ):
        vt_dev = next(self.parameters()).device
        images = images.to(vt_dev, non_blocking=True)

        v = self.encode_images(images, **token_ft_kwargs)                # (B, D)
        t = self.encode_texts(text_input_ids, text_attention_mask)       # (M, D)
        v32 = self._to_fp32(v)
        t32 = self._to_fp32(t)
        temp = self.log_temp.exp().clamp(1e-6, 100.0)
        logits = (v32 @ t32.t()) / temp                                  # (B, M)

        out: Dict[str, Any] = {
            "logits": logits,
            "temp": temp,
        }
        if targets is None and (image_label_lists is not None) and (text_labels is not None):
            if isinstance(text_labels, torch.Tensor):
                text_labels = text_labels.detach().cpu().tolist()
            targets = self._build_targets_from_labels(
                image_label_lists=image_label_lists,
                text_labels=text_labels,
                device=logits.device,
                dtype=logits.dtype,
            )
        if targets is not None:
            targets = targets.to(vt_dev, non_blocking=True).float()

            if self.use_asl:
                loss = self._asymmetric_loss(logits, targets)
            else:
                pos = targets.sum()
                neg = targets.numel() - pos
                if pos > 0:
                    pos_weight = (neg / pos).clamp(min=1.0)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, targets,
                        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, targets)

            out["loss"] = loss

        if return_embeddings:
            out["vision_emb"] = v
            out["text_emb"] = t

        return out