# train.py
import os, time, argparse
import torch
from torch.utils.data import DataLoader

# --- 학습 스텝/데이터셋 임포트 ---
from llm_machine import train_step_linked
from llm_machine.data_linked import (
    ImageDatasetMultiLabel,
    ImageCollatorMulti,
)

from main import (
    build_model, load_jsonl, build_label2texts, build_text_batch_one_per_label,
    build_train_transform, format_eta, save_weights,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_img", type=str, required=True)
    ap.add_argument("--items_txt", type=str, required=True)
    ap.add_argument("--ckpt_dir",  type=str, required=True)

    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size_img", type=int, default=2)
    ap.add_argument("--num_workers_img", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=10)

    # --- 재개 옵션 ---
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_tag", type=str, default="")
    ap.add_argument("--resume_all", type=int, default=0)

    # --- 체크포인트 로드 옵션 ---
    ap.add_argument("--vt_ckpt_path", type=str, default="/home/piai/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch056_FINAL.pt")
    ap.add_argument("--token_ckpt_path", type=str, default="/home/piai/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/epoch100_new_pattern.pth")
    ap.add_argument("--strict_load", type=int, default=0)
    ap.add_argument("--save_full_state", type=int, default=1)

    # --- 손실 가중치 (feats-text bag) ---
    ap.add_argument("--alpha_ft", type=float, default=1.0)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = (device == "cuda")

    # 1) 모델 로드
    vt, token_model = build_model(
        device,
        vt_ckpt_path=(args.vt_ckpt_path or None),
        token_ckpt_path=(args.token_ckpt_path or None),
        strict_load=bool(args.strict_load)
    )

    # 2) 데이터 로드
    items_img = load_jsonl(args.items_img)  # image_path, labels 포함
    items_txt = load_jsonl(args.items_txt)
    label2texts = build_label2texts(items_txt)

    tfm = build_train_transform()
    img_ds = ImageDatasetMultiLabel(items_img, image_transform=tfm)
    collate_fn = ImageCollatorMulti()

    img_loader = DataLoader(
        img_ds,
        batch_size=args.batch_size_img,
        shuffle=True,
        num_workers=args.num_workers_img,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers_img > 0),
    )

    steps_per_epoch = len(img_loader)

    # 3) 옵티마이저/AMP
    optimizer = torch.optim.AdamW(
        [p for p in vt.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 4) 재개(선택)
    start_epoch, global_step = 1, 0
    if args.resume_path and os.path.isfile(args.resume_path):
        raw = torch.load(args.resume_path, map_location=device)
        if raw.get("vt"):
            vt.load_state_dict(raw["vt"], strict=False)
        if raw.get("token"):
            (vt.token_model.module if hasattr(vt.token_model, "module") else vt.token_model).load_state_dict(
                raw["token"], strict=False
            )
        if args.resume_all and raw.get("optimizer"):
            optimizer.load_state_dict(raw["optimizer"])
        if args.resume_all and raw.get("scaler") and scaler is not None:
            scaler.load_state_dict(raw["scaler"])
        start_epoch = int(raw.get("epoch", 0)) + 1
        global_step = int(raw.get("global_step", 0))
        print(f"[Resume] from file: epoch={start_epoch-1}, global_step={global_step}")
    elif args.resume_tag:
        p = os.path.join(args.ckpt_dir, f"train_state_{args.resume_tag}.pt")
        if os.path.isfile(p):
            raw = torch.load(p, map_location=device)
            if raw.get("vt"):
                vt.load_state_dict(raw["vt"], strict=False)
            if raw.get("token"):
                (vt.token_model.module if hasattr(vt.token_model, "module") else vt.token_model).load_state_dict(
                    raw["token"], strict=False
                )
            if args.resume_all and raw.get("optimizer"):
                optimizer.load_state_dict(raw["optimizer"])
            if args.resume_all and raw.get("scaler") and scaler is not None:
                scaler.load_state_dict(raw["scaler"])
            start_epoch = int(raw.get("epoch", 0)) + 1
            global_step = int(raw.get("global_step", 0))
            print(f"[Resume] from tag: epoch={start_epoch-1}, global_step={global_step}")
        else:
            print(f"[Resume] not found tag at {p}")

    # 5) 학습 루프
    for epoch in range(start_epoch, args.epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        ep_t0 = time.perf_counter()

        # --- 에포크별 누적 통계 (loss) ---
        ep_loss_sum = 0.0
        ep_ft_loss_sum = 0.0
        ep_loss_cnt = 0  # batch 카운트

        for step, batch_img in enumerate(img_loader, start=1):
            t0 = time.perf_counter()

            # 1) 배치에서 등장한 라벨별 텍스트 1개씩 샘플링
            batch_txt = build_text_batch_one_per_label(
                batch_img=batch_img,
                label2texts=label2texts,
                tokenizer=vt.text_encoder.tokenizer,
                max_length=64,
            )
            if batch_txt is None:
                continue

            # 2) 한 스텝 학습 (alpha_ft 가중치 전달)
            logs = train_step_linked(
                vt, batch_img, batch_txt, optimizer, scaler,
                alpha_ft=args.alpha_ft
            )

            # --- 손실 값 추출 ---
            total_loss = float(logs.get("loss", float("nan")))
            ft_loss = logs.get("feat_text_loss", None)
            ft_loss_val = float(ft_loss) if ft_loss is not None else float("nan")

            ep_loss_sum += total_loss
            ep_loss_cnt += 1
            if not torch.isnan(torch.tensor(ft_loss_val)):
                ep_ft_loss_sum += ft_loss_val

            global_step += 1
            iter_time = time.perf_counter() - t0

            # --- 로그 출력 ---
            if (step % args.log_every) == 0:
                # 에포크 평균 loss
                avg_loss = ep_loss_sum / max(ep_loss_cnt, 1)
                avg_ft_loss = ep_ft_loss_sum / max(ep_loss_cnt, 1)

                max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
                lr_cur = optimizer.param_groups[0]["lr"]
                steps_done = (epoch - 1) * steps_per_epoch + step
                eta_sec = (args.epochs * steps_per_epoch - steps_done) * iter_time

                # 손실 문자열 구성
                loss_str = f"total {total_loss:.4f} (ep {avg_loss:.4f})"
                if ft_loss is not None:
                    loss_str += f" | ft {ft_loss_val:.4f} (ep {avg_ft_loss:.4f})"

                print(
                    f">> Train Epoch: [{epoch}] [{step}/{steps_per_epoch}] "
                    f"eta: {format_eta(eta_sec)} "
                    f"Loss: {loss_str} "
                    f"iter: {iter_time:.4f}s lr: {lr_cur:.2e} max mem: {int(max_mem_mb)} MB"
                )

        # 에포크 종료 로그
        ep_time = time.perf_counter() - ep_t0
        print(f">> Epoch [{epoch}] done in {format_eta(ep_time)}")

        # 4 에포크마다 체크포인트 저장
        if epoch % 4 == 0:
            save_weights(
                vt, vt.token_model, optimizer, scaler,
                epoch, global_step, args.ckpt_dir,
                tag=f"epoch{epoch:03d}", save_full_state=bool(args.save_full_state)
            )

    # 마지막 체크포인트
    save_weights(
        vt, vt.token_model, optimizer, scaler,
        args.epochs, global_step, args.ckpt_dir,
        tag="last", save_full_state=bool(args.save_full_state)
    )


if __name__ == "__main__":
    main()
