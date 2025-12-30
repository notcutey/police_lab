import torch
import torch.nn.functional as F
from networks import Token
@torch.no_grad()
def _minmax_norm(t: torch.Tensor, eps: float = 1e-6):
    tmin = t.amin(dim=-1, keepdim=True)
    tmax = t.amax(dim=-1, keepdim=True)
    return (t - tmin) / (tmax - tmin + eps)

def explain_with_grad_eclip(model: Token,
                            img: torch.Tensor,          # [B, 2048, H, W]
                            text_vec: torch.Tensor,     # [B, 1024]
                            up_hw: tuple = None):       # 최종 업샘플 크기 (H_img, W_img)
    """
    반환:
      score_map: [B, N_k]  (리쉐이프 전)
      heatmap:   [B, Hk, Wk] (리쉐이프 후, up_hw 지정 시 업샘플)
      sim:       [B]  (cosine similarity)
    """
    model.eval()
    B = img.size(0)

    # 1) forward + aux 저장
    proj_out, aux = model(img, return_aux=True)   # proj_out: [B, 1024]
    proj_out = proj_out.requires_grad_(True)
    proj_out.retain_grad()

    # 2) cosine similarity S
    F_I = F.normalize(proj_out, dim=-1)           # [B, 1024]
    F_T = F.normalize(text_vec, dim=-1)           # [B, 1024]
    sim = (F_I * F_T).sum(dim=-1)                 # [B]

    # 3) backward to get channel weights w
    loss = sim.sum()
    loss.backward(retain_graph=True)
    w = proj_out.grad.clone()                     # [B, 1024]
    w = F.relu(w)                                 # 음수 억제
    # L1 normalize (선택)
    w = w / (w.norm(p=1, dim=-1, keepdim=True) + 1e-6)

    # 4) value-like feature map p_i 만들기
    x_mem = aux["x_mem"]                          # [B, N_k, mid_dim]
    attn_map = aux["last_attn_map"]               # [B, H, N_q, N_k]
    B, N_k, mid_dim = x_mem.shape
    _, Hh, N_q, _ = attn_map.shape

    # (a) 객체별 proj weight 슬라이스
    W_list = model._slice_proj_per_object()       # len=N_q, 각 [1024, mid_dim]

    # (b) head 평균 → [B, N_q, N_k]
    attn_avg = attn_map.mean(dim=1)               # [B, N_q, N_k]

    # (c) 느슨한 공간 가중치 lambda: query 평균 후 min-max norm → [B, N_k]
    lam = _minmax_norm(attn_avg.mean(dim=1))      # [B, N_k], 0~1

    # (d) p: 위치별 1024-D 벡터 만들기
    #     p[b, i, :] = sum_o attn_avg[b,o,i] * ( W_o @ x_mem[b,i,:] )
    p = torch.zeros(B, N_k, 1024, device=x_mem.device, dtype=x_mem.dtype)
    for o in range(N_q):
        W_o = W_list[o]                           # [1024, mid_dim]
        # x_proj: [B, N_k, 1024]
        x_proj = torch.matmul(x_mem, W_o.t())
        # 가중합 누적
        a = attn_avg[:, o, :].unsqueeze(-1)       # [B, N_k, 1]
        p += a * x_proj                           # [B, N_k, 1024]

    # 5) 위치별 점수: score_i = lambda_i * <p_i, w>
    #    (배치별 w와 브로드캐스트)
    score = (p * w.unsqueeze(1)).sum(dim=-1)      # [B, N_k]
    score = lam * score                           # [B, N_k]

    # 6) 리쉐이프 → (Hk, Wk), 업샘플
    #    N_k = H*W (여기선 conv 출력의 H*W)
    #    conv에서 썼던 H,W를 그대로 재현하려면 model.forward에서 기록해도 됨
    #    여기선 역으로 정사각 격자 가정 또는 호출자가 넘겨주도록
    HkWk = int(N_k ** 0.5)
    if HkWk * HkWk == N_k:
        Hk, Wk = HkWk, HkWk
    else:
        # conv의 출력 H,W를 아는 경우 그걸 써도 됨
        Hk, Wk = 16, N_k // 16  # 안전용 기본값

    heat = score.view(B, 1, Hk, Wk)               # [B,1,Hk,Wk]
    heat = _minmax_norm(heat.flatten(1)).view_as(heat)  # 0~1 정규화

    if up_hw is not None:
        heat_up = F.interpolate(heat, size=up_hw, mode="bilinear", align_corners=False)
    else:
        heat_up = heat

    return score, heat_up.squeeze(1), sim.detach()
