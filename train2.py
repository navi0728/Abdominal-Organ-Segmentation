import math, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from networks import ConvNeXtUNet
from utils.dataset_trans import WORDdataset

NUM_CLASSES = 17

def poly_lr_scheduler(optimizer, init_lr, cur_iter, max_iter, power=0.9):
    lr = init_lr * (1 - cur_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda", desc="train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, n_batches = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(is_train):
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True, dtype=torch.long)
            imgs = imgs.repeat(1, 3, 1, 1) if imgs.shape[1] == 1 else imgs

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}")

    return running_loss / max(1, n_batches)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ct_tf = transforms.Compose([transforms.Lambda(lambda t: (t - 0.5) / 0.5)])
    train_ds = WORDdataset(args.data_root, mode="train", ct_transform=ct_tf)
    val_ds = WORDdataset(args.data_root, mode="val"  , ct_transform=ct_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = ConvNeXtUNet("convnext_tiny.fb_in22k_ft_in1k",
                  num_classes=NUM_CLASSES,
                  encoder_pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    best_val = math.inf
    global_iter = 0
    max_iter = args.epochs * len(train_loader)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        poly_lr_scheduler(optimizer, args.lr, global_iter, max_iter)
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, device, desc="train")
        global_iter += len(train_loader)

        with torch.no_grad():
            val_loss = run_epoch(val_loader, model, loss_fn, None, device, desc="val")

        print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        ckpt_dir = Path(args.ckpt_dir);  ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "val_loss": val_loss},
                   ckpt_dir / f"epoch{epoch:03d}.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print("  → best model updated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt2/dataset/WORD/minju/WORD-slices")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_word")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args)