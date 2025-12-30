# vt_siglip/train_utils_linked.py
import torch
from .siglip_model import VisionTextSigLIP
from .data_linked import ImageBatchMulti, TextBatchSingle, build_targets_imgmulti_textsingle


def train_step_linked(
    model: VisionTextSigLIP,
    token_model,                      # ✅ 새로 추가: 외부 Token 모델
    batch_img: ImageBatchMulti,
    batch_txt: TextBatchSingle,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    ✅ Token(vision)과 VT(text+proj)를 완전히 분리한 학습 스텝.
    - Token: forward_test 로 image_feats 추출 (no_grad + frozen)
    - VT: image_feats + text 로 contrastive / ASL 학습
    """
    model.train()
    device = next(model.parameters()).device

    # 1) 이미지 텐서를 Token 디바이스로 이동
    images = batch_img.images.to(device, non_blocking=True)

    # 2) Token 모델로 feature 추출 (학습/grad 없음)
    with torch.no_grad():
        image_feats, _ = token_model.forward_test(images)    # (B, Dv)

    # 3) 멀티라벨 타깃 생성 (B, M)
    targets = build_targets_imgmulti_textsingle(batch_img.label_sets, batch_txt.labels).to(
        device
    )

    # 텍스트 텐서는 LLMTextEncoder 내부에서 실제 디바이스로 옮겨지지만,
    # 여기서는 CPU 텐서를 그대로 넣어도 됨.
    input_ids = batch_txt.input_ids
    attention_mask = batch_txt.attention_mask

    optimizer.zero_grad(set_to_none=True)

    use_amp = (scaler is not None) and (device.type == "cuda")
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            out = model(image_feats, input_ids, attention_mask, targets=targets)
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        out = model(image_feats, input_ids, attention_mask, targets=targets)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

    # "temperature" 또는 "temp" 키 모두 지원
    temp_tensor = out.get("temperature", out.get("temp", None))
    if temp_tensor is None:
        temp_tensor = model.log_temp.exp()

    return {
        "loss": float(loss.detach().cpu().item()),
        "temp": float(temp_tensor.detach().cpu().item()),
    }
