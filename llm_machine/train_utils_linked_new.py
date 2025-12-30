# vt_siglip/train_utils_linked.py
import torch
from .siglip_model import VisionTextSigLIP
from .data_linked import ImageBatchMulti, TextBatchSingle, build_targets_imgmulti_textsingle


# ===================== #
#  train_step_linked    #
# ===================== #

def train_step_linked(
    vt: VisionTextSigLIP,
    token_model,                      # ✅ 외부 Token 모델 (freeze 상태)
    batch_img: ImageBatchMulti,
    batch_txt: TextBatchSingle,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    이미지 → Token.forward_test → image_feats
    image_feats + 텍스트(batch_txt)를 vt에 넣어서
    vt 내부의 contrastive(v, h_attn) loss로 학습하는 한 스텝.
    """
    vt.train()
    device = next(vt.parameters()).device
    use_amp = (device.type == "cuda")

    # 1) 배치 텐서 디바이스 이동
    if hasattr(batch_img, "images"):
        images = batch_img.images.to(device, non_blocking=True)
    elif isinstance(batch_img, dict) and "images" in batch_img:
        images = batch_img["images"].to(device, non_blocking=True)
    else:
        raise TypeError("batch_img에서 images 텐서를 찾을 수 없다.")

    input_ids = batch_txt.input_ids.to(device, non_blocking=True)
    attention_mask = batch_txt.attention_mask.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    # 2) AMP context
    ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
    with ctx:
        # Token은 freeze 상태이므로 grad 불필요
        with torch.no_grad():
            global_vector, token_num = token_model.forward_test(images)  # (B, Dv)
        targets = build_targets_imgmulti_textsingle(batch_img.label_sets, batch_txt.labels).to(device)


        # 🔥 vt.forward 가 v / h_attn / contrastive loss 를 모두 처리한다고 가정
        out = vt(
            image_feats=token_num,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            targets=targets,
            return_embeddings=False,
        )

        loss = out["loss"]

    # 3) backward + optimizer step
    if scaler is not None and use_amp:

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    logs = {
        "loss": float(loss.detach().cpu().item()),
        "temp": float(out.get("temp", 0.0)),
    }
    return logs
