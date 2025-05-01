import math, argparse
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import os

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.parse import parse_args
from utils.dataset_ver2 import WORDdataset


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda", desc="train"):

    # loss 평균 구하기 위해 필요한 변수
    running_loss, n_batches = 0.0, 0

    if optimizer is not None:
        model.train().to(device)
        grad_enabled = True  
    else:
        model.eval().to(device)
        grad_enabled = False 
    
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(grad_enabled):
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.long()

            predict = model(imgs)
            loss = loss_fn(predict, labels)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}")

    return running_loss / max(1, n_batches)

def main(args):

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ct_tf = transforms.Compose([
                        transforms.Lambda(lambda t: (t - 0.5) / 0.5),     
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.RandomHorizontalFlip(p=0.5),
                        ])
    
    train_ds = WORDdataset(args.data_root, mode="train", ct_transform=ct_tf)
    val_ds = WORDdataset(args.data_root, mode="val", ct_transform=ct_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = smp.Unet(
                    encoder_name="resnet34",       
                    encoder_weights="imagenet",    
                    in_channels=1,                 
                    classes=17                     
                    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    best_val = math.inf

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, device, desc="train")

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
    
    args = parser.parse_args()
    main(args)



    