import math
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
    """
    Class-Balanced Weight 계산 (변경 없음)
    """
    if device is None:
        device = (
            label_pos_counts.device
            if isinstance(label_pos_counts, torch.Tensor)
            else torch.device("cpu")
        )

    counts = torch.as_tensor(
        label_pos_counts, dtype=torch.float32, device=device
    ).clamp(min=1.0)
    num = 1.0 - beta
    den = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32, device=device), counts)
    w = num / torch.clamp(den, min=1e-12)

    # 평균 1 근처로 스케일링
    w = w * (w.numel() / w.sum().clamp(min=1e-12))

    # 클램핑
    w = torch.clamp(w, min=w_min, max=w_max)
    return w


class TextSelfAttentionBlock(nn.Module):
    """
    텍스트 임베딩용 Transformer 블록
    - Self-Attention + FFN + LayerNorm + Residual
    - 입력/출력: (M, D) 또는 (B, M, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # (B, M, D) 포맷 사용
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (M, D) 또는 (B, M, D)
        """
        squeeze_batch = False
        if x.dim() == 2:
            # (M, D) -> (1, M, D)
            x = x.unsqueeze(0)
            squeeze_batch = True

        # Self-Attention + Residual
        x_ln = self.ln1(x)
        attn_out, _ = self.self_attn(x_ln, x_ln, x_ln)  # (B, M, D)
        x = x + attn_out

        # FFN + Residual
        x_ln2 = self.ln2(x)
        x = x + self.mlp(x_ln2)

        if squeeze_batch:
            x = x.squeeze(0)  # 다시 (M, D)

        return x


class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention (raw score만 반환)
    - 입력:
        q: (B, Nq, D)
        k: (B, Nk, D)
        v: (B, Nk, D)   # 호환성 위해 받지만 현재는 사용하지 않음
    - 출력:
        scores_mean: (B, Nq, Nk)  # head 평균 raw attention score
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,

    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # 정의만, 현재 사용 X

    def forward(
        self,
        q: torch.Tensor,   # (B, Nq, D)
        k: torch.Tensor,   # (B, Nk, D)
        v: torch.Tensor,   # (B, Nk, D)
    ) -> torch.Tensor:
        B, Nq, D = q.shape
        _, Nk, _ = k.shape
        H = self.num_heads
        Dh = D // H

        # Q, K projection + head 분리
        q = self.q(q).reshape(B, Nq, H, Dh).permute(0, 2, 1, 3)   # (B, H, Nq, Dh)
        k = self.k(k).reshape(B, Nk, H, Dh).permute(0, 2, 1, 3)   # (B, H, Nk, Dh)

        # scaled dot-product (softmax 없음)
        scores = (q @ k.transpose(-2, -1)) * self.scale           # (B, H, Nq, Nk)

        # head 평균
        scores_mean = scores.mean(dim=1)                          # (B, Nq, Nk)
        return scores_mean


class VisionTextSigLIP(nn.Module):

    def __init__(
        self,
        text_encoder: LLMTextEncoder,
        vision_dim: int = 1024,
        proj_out_dim: int = 1024,
        temperature_init: float = 1.0,
        llm_hidden_size: Optional[int] = None,
        sync_text_device: bool = True,

        # --- Vision projector 옵션 ---
        use_vision_proj: bool = True,
        vision_proj_type: str = "linear",   # "linear" | "mlp"
        vision_proj_hidden: Optional[int] = None,
        use_vision_residual: bool = False,
        vision_dropout: float = 0.0,

        # --- Cross-Attention 옵션 ---
        use_xattn: bool = True,
        xattn_dropout: float = 0.0,

        # --- Asymmetric Loss (멀티라벨) 옵션 ---
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

        self.text_encoder = text_encoder
        self.tokenizer = getattr(text_encoder, "tokenizer", None)

        if llm_hidden_size is None:
            llm_hidden_size = self.text_encoder.detect_hidden_size()

        self.vision_dim = vision_dim
        self.proj_out_dim = proj_out_dim

        # ───────── Text projector ─────────
        self.text_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, proj_out_dim),
            nn.LayerNorm(proj_out_dim),
        )

        # ───────── Vision projector ─────
        self.use_vision_proj = use_vision_proj
        self.use_vision_residual = use_vision_residual

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

        # ───────── Cross-Attention 설정 ─────────
        self.use_xattn = use_xattn
        self.xattn_dropout = xattn_dropout
        if self.use_xattn:
            self.cross_attn = CrossAttention(
                dim=proj_out_dim,
                num_heads=8,
                qkv_bias=True
            )
        else:
            self.cross_attn = None

        # ───────── Text Self-Attention Block ─────────
        # 문양 텍스트 프로토타입에 대해 Self-Attn + FFN + LayerNorm + Residual 수행
        self.use_text_self_attn = True
        if self.use_text_self_attn:
            self.text_self_attn_block = TextSelfAttentionBlock(
                dim=proj_out_dim,
                num_heads=8,
                mlp_ratio=4.0,
                dropout=xattn_dropout,
            )
        else:
            self.text_self_attn_block = None

        # ───────── Temperature 파라미터 ─────────
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor(temperature_init, dtype=torch.float32))
        )

        # ───────── Text encoder 디바이스 동기화 ─────────
        if sync_text_device:
            vt_dev = next(self.parameters()).device
            try:
                self.text_encoder.to(vt_dev)
                if hasattr(self.text_encoder, "device"):
                    self.text_encoder.device = (
                        vt_dev.type if isinstance(vt_dev, torch.device) else str(vt_dev)
                    )
            except Exception:
                pass

        # ───────── Asymmetric Loss & Class-Balanced Weight ─────────
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

    # ==============================
    #   Helper / encode 함수들
    # ==============================

    @torch.no_grad()
    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
            self.use_cbw = False
        else:
            dev = (
                self.class_weights.device
                if self.class_weights is not None
                else next(self.parameters()).device
            )
            self.class_weights = class_weights.float().to(dev)
            self.use_cbw = True

    @torch.no_grad()
    def _to_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    # --------- (중요) Vision 토큰 4개까지 처리하도록 수정 ---------
    def _project_image_feats(self, image_feats: torch.Tensor) -> torch.Tensor:
        """
        image_feats:
            - (B, Dv) : 기존 single global vector
            - (B, Nq, Dv) : 한 이미지당 Nq개 토큰 (지금은 Nq=4)
        반환:
            - (B, D) 또는 (B, Nq, D)  (proj_out_dim = D)
        """
        v = image_feats

        if self.use_vision_proj and (self.vision_proj is not None):
            # target dtype
            if isinstance(self.vision_proj, nn.Sequential):
                first_linear = next(m for m in self.vision_proj if isinstance(m, nn.Linear))
                target_dtype = first_linear.weight.dtype
            else:
                target_dtype = self.vision_proj.weight.dtype
            v = v.to(target_dtype)

            v_in = v  # residual용 저장

            if v.dim() == 2:
                # (B, Dv)
                v = self.vision_proj(v)  # (B, D)
            elif v.dim() == 3:
                # (B, Nq, Dv) -> (B*Nq, Dv) -> proj -> (B, Nq, D)
                B, Nq, Dv = v.shape
                v = v.reshape(B * Nq, Dv)
                v = self.vision_proj(v)               # (B*Nq, D)
                v = v.reshape(B, Nq, self.proj_out_dim)
            else:
                raise ValueError(f"Unexpected image_feats dim: {v.shape}")

            if self.use_vision_residual:
                if self.vision_residual is not None:
                    if v_in.dim() == 2:
                        v = v + self.vision_residual(v_in)
                    elif v_in.dim() == 3:
                        B, Nq, Dv = v_in.shape
                        res = v_in.reshape(B * Nq, Dv)
                        res = self.vision_residual(res)      # (B*Nq, D)
                        res = res.reshape(B, Nq, self.proj_out_dim)
                        v = v + res
                else:
                    v = v + v_in

        # 필요하면 normalize 가능
        # v = F.normalize(v, dim=-1)
        return v

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        t_raw = self.text_encoder.encode_text(input_ids, attention_mask)  # (M, H_lm)
        t_raw = t_raw.to(next(self.text_proj.parameters()).dtype)
        t = self.text_proj(t_raw)                                         # (M, D)
        # 마찬가지로 필요시 normalize
        # t = F.normalize(t, dim=-1)
        return t

    # ==============================
    #   Target 빌더 (라벨 → (B,M) 원핫)
    # ==============================
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

    # ==============================
       #   Asymmetric Loss (멀티라벨)
    # ==============================
    def _asymmetric_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.asl_eps

        x_sigmoid = torch.sigmoid(logits)          # p
        xs_pos = x_sigmoid                         # p
        xs_neg = 1.0 - x_sigmoid                   # 1-p

        if self.asl_clip is not None and self.asl_clip > 0:
            xs_neg = (xs_neg + self.asl_clip).clamp(max=1.0)

        log_pos = torch.log(xs_pos.clamp(min=eps))
        log_neg = torch.log(xs_neg.clamp(min=eps))

        gamma_pos = self.asl_gamma_pos
        gamma_neg = self.asl_gamma_neg

        pos_weight = torch.pow(1.0 - xs_pos, gamma_pos)
        neg_weight = torch.pow(1.0 - xs_neg, gamma_neg)

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

    # ==============================
    #   Cross-Attention (이미지 토큰 → 문양 텍스트)
    # ==============================
    def _cross_attend_image_to_text(
        self,
        v_tokens: torch.Tensor,  # (B, Nq, D)
        t_txt: torch.Tensor,     # (M, D)
    ) -> torch.Tensor:
        """
        Q = 이미지 토큰들 [B, Nq, D]
        K,V = 문양 텍스트 [B, M, D] (B 방향 broadcast)

        반환:
        - scores: (B, M)  # Nq 토큰 중 최대값 기반 max pooling 사용
        """
        B, Nq, D = v_tokens.shape
        M = t_txt.shape[0]

        q = v_tokens                               # (B, Nq, D)
        k = t_txt.unsqueeze(0).expand(B, M, D)     # (B, M, D)
        v = k                                      # 인터페이스용

        attn_scores = self.cross_attn(q, k, v)      # (B, Nq, M)

        # ★ 평균 ➜ max pooling (MIL 구조)
        scores = attn_scores.mean(dim=1)          # (B, M)

        return scores


    # ==============================
    #   Forward
    # ==============================
    def forward(
        self,
        image_feats: torch.Tensor,         # (B, Dv) 또는 (B, Nq, Dv)  ← 지금은 (B,4,1024)
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        image_label_lists: Optional[Sequence[Sequence[Hashable]]] = None,
        text_labels: Optional[Sequence[Hashable]] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:

        vt_dev = next(self.parameters()).device
        image_feats = image_feats.to(vt_dev, non_blocking=True)

        # 1) 이미지 토큰 임베딩 v_tokens
        v = self._project_image_feats(image_feats)       # (B, D) or (B, Nq, D)

        if v.dim() == 2:
            # 기존 single vector 구조도 호환
            v_tokens = v.unsqueeze(1)                    # (B, 1, D)
        else:
            v_tokens = v                                 # (B, Nq, D)  ← Nq=4

        # 2) 문양 텍스트 임베딩 프로토타입 t
        t = self.encode_texts(text_input_ids, text_attention_mask)  # (M, D)

        # 2-1) Text Self-Attention + FFN + LayerNorm + Residual
        if getattr(self, "use_text_self_attn", False) and (self.text_self_attn_block is not None):
            t = self.text_self_attn_block(t)  # (M, D) -> (M, D)

        # 3) cross-attention raw score → logits
        if self.use_xattn and (self.cross_attn is not None):
            scores = self._cross_attend_image_to_text(v_tokens, t)   # (B, M)
        else:
            # xattn 끄면 그냥 점곱 사용 (옵션)
            # v_tokens: (B, Nq, D), t: (M, D)
            B, Nq, D = v_tokens.shape
            M = t.shape[0]
            # (B, Nq, D) x (D, M) -> (B, Nq, M)
            dot = torch.matmul(v_tokens, t.t().unsqueeze(0).expand(B, D, M))
            scores, _ = dot.mean(dim=1)
        
        temp = self.log_temp.exp().clamp(1e-6, 100.0)
        logits = scores / temp                                       # (B, M)

        out: Dict[str, Any] = {
            "logits": logits,
            "temp": temp,
        }

        # 4) target 없으면 label 리스트로부터 생성
        if targets is None and (image_label_lists is not None) and (text_labels is not None):
            if isinstance(text_labels, torch.Tensor):
                text_labels = text_labels.detach().cpu().tolist()
            targets = self._build_targets_from_labels(
                image_label_lists=image_label_lists,
                text_labels=text_labels,
                device=logits.device,
                dtype=logits.dtype,
            )

        # 5) Loss (ASL or BCEWithLogits)
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
                        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, targets)

            out["loss"] = loss

        if return_embeddings:
            out["vision_emb"] = v_tokens    # (B, Nq, D)  ← 토큰 단위 임베딩
            out["text_emb"] = t             # (M, D)

        return out
