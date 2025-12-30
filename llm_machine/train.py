# train.py
import os, time, argparse
import torch
from torch.utils.data import DataLoader

from llm_machine import train_step_linked
from llm_machine.data_linked import ImageDatasetMultiLabel, ImageCollatorMulti

from main import (
    build_model, load_jsonl, build_label2texts, build_text_batch_one_per_label,
    compute_label_hit_ratio_at_k, build_train_transform, format_eta, save_weights,
)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_img", type=str, required=True)
    ap.add_argument("--items_txt", type=str, required=True)
    ap.add_argument("--ckpt_dir",  type=str, required=True)

    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size_img", type=int, default=128)
    ap.add_argument("--num_workers_img", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_hit_k", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_tag", type=str, default="")
    ap.add_argument("--resume_all", type=int, default=0)

    ap.add_argument("--vt_ckpt_path", type=str, default="")
    ap.add_argument("--token_ckpt_path", type=str, default="")
    ap.add_argument("--strict_load", type=int, default=0)
    ap.add_argument("--save_full_state", type=int, default=1)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = (device == "cuda")

    # 1) 모델
    vt, token_model = build_model(
        device,
        vt_ckpt_path=(args.vt_ckpt_path or None),
        token_ckpt_path=(args.token_ckpt_path or None),
        strict_load=bool(args.strict_load)
    )

    # 2) 데이터
    items_img = load_jsonl(args.items_img)
    items_txt = load_jsonl(args.items_txt)
    label2texts = build_label2texts(items_txt)

    tfm = build_train_transform()
    img_ds = ImageDatasetMultiLabel(items_img, image_transform=tfm)
    img_loader = DataLoader(
        img_ds, batch_size=args.batch_size_img, shuffle=True, num_workers=args.num_workers_img,
        collate_fn=ImageCollatorMulti(), pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers_img > 0)
    )

    steps_per_epoch = len(img_loader)
    total_steps = args.epochs * steps_per_epoch

    # 3) 옵티마이저/AMP
    optimizer = torch.optim.AdamW([p for p in vt.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 4) 재개(선택)
    start_epoch, global_step = 1, 0
    if args.resume_path and os.path.isfile(args.resume_path):
        raw = torch.load(args.resume_path, map_location=device)
        if raw.get("vt"): vt.load_state_dict(raw["vt"], strict=False)
        if raw.get("token"): (vt.token_model.module if hasattr(vt.token_model, "module") else vt.token_model).load_state_dict(raw["token"], strict=False)
        if args.resume_all and raw.get("optimizer"): optimizer.load_state_dict(raw["optimizer"])
        if args.resume_all and raw.get("scaler") and scaler is not None: scaler.load_state_dict(raw["scaler"])
        start_epoch = int(raw.get("epoch", 0)) + 1
        global_step = int(raw.get("global_step", 0))
        print(f"[Resume] from file: epoch={start_epoch-1}, global_step={global_step}")
    elif args.resume_tag:
        p = os.path.join(args.ckpt_dir, f"train_state_{args.resume_tag}.pt")
        if os.path.isfile(p):
            raw = torch.load(p, map_location=device)
            if raw.get("vt"): vt.load_state_dict(raw["vt"], strict=False)
            if raw.get("token"): (vt.token_model.module if hasattr(vt.token_model, "module") else vt.token_model).load_state_dict(raw["token"], strict=False)
            if args.resume_all and raw.get("optimizer"): optimizer.load_state_dict(raw["optimizer"])
            if args.resume_all and raw.get("scaler") and scaler is not None: scaler.load_state_dict(raw["scaler"])
            start_epoch = int(raw.get("epoch", 0)) + 1
            global_step = int(raw.get("global_step", 0))
            print(f"[Resume] from tag: epoch={start_epoch-1}, global_step={global_step}")
        else:
            print(f"[Resume] not found tag at {p}")

    # 5) 학습 루프
    for epoch in range(start_epoch, args.epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        cum_hits = 0
        cum_gt   = 0
        ep_t0 = time.perf_counter()

        for step, batch_img in enumerate(img_loader, start=1):
            t0 = time.perf_counter()

            batch_txt = build_text_batch_one_per_label(
                batch_img=batch_img,
                label2texts=label2texts,
                tokenizer=vt.text_encoder.tokenizer,
                max_length=64
            )
            if batch_txt is None:
                continue

            # 1) 한 스텝 학습
            logs = train_step_linked(vt, batch_img, batch_txt, optimizer, scaler)
            cur_loss = float(logs.get("arcface_loss", logs.get("loss", float("nan"))))

            # 2) 라벨 히트@K 측정
            text_label_ids = getattr(batch_txt, "label_ids", None)
            batch_ratio = float("nan")
            if text_label_ids is not None:
                img_label_sets = [set(map(int, labs)) for labs in batch_img.label_sets]
                device_t = next(vt.parameters()).device
                images_t = batch_img.images.to(device_t, non_blocking=True)
                input_ids = batch_txt.input_ids.to(device_t, non_blocking=True)
                attention_mask = batch_txt.attention_mask.to(device_t, non_blocking=True)
                ratio, hits, gt = compute_label_hit_ratio_at_k(
                    vt=vt,
                    images_t=images_t,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_label_sets=img_label_sets,
                    text_label_ids=text_label_ids,
                    k=args.label_hit_k,
                )
                batch_ratio = ratio
                cum_hits += hits
                cum_gt   += gt

            global_step += 1
            iter_time = time.perf_counter() - t0

            if (step % args.log_every) == 0:
                cum_ratio = (cum_hits / cum_gt * 100.0) if cum_gt > 0 else float("nan")
                max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
                lr_cur = optimizer.param_groups[0]["lr"]
                steps_done = (epoch - 1) * steps_per_epoch + step
                eta_sec = (args.epochs * steps_per_epoch - steps_done) * iter_time
                print(
                    f">> Train Epoch: [{epoch}] [{step}/{steps_per_epoch}] "
                    f"eta: {format_eta(eta_sec)} "
                    f"ArcFace loss: {cur_loss:.4f} "
                    f"Label-Hit@{args.label_hit_k}: batch {batch_ratio:6.3f}% | epoch {cum_ratio:6.3f}% "
                    f"iter: {iter_time:.4f}s lr: {lr_cur:.2e} max mem: {int(max_mem_mb)} MB"
                )

        print(f">> Epoch [{epoch}] done in {format_eta(time.perf_counter() - ep_t0)}")

        if epoch % 4 == 0:
            save_weights(vt, vt.token_model, optimizer, scaler, epoch, global_step, args.ckpt_dir, tag=f"epoch{epoch:03d}", save_full_state=bool(args.save_full_state))

    save_weights(vt, vt.token_model, optimizer, scaler, args.epochs, global_step, args.ckpt_dir, tag="last", save_full_state=bool(args.save_full_state))

if __name__ == "__main__":
    main()
