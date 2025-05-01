import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils.dataset_ver2 import WORDdataset

NUM_CLASSES = 17

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


## COCO 색상 팔레트
PALETTE = [
    (0,   0,   0  ),  # 0 background
    (255, 0,   0  ),  # 1
    (0,   255, 0  ),  # 2
    (0,   0,   255),  # 3
    (255, 255, 0  ),  # 4
    (255, 0,   255),  # 5
    (0,   255, 255),  # 6
    (128, 0,   0  ),  # 7
    (0,   128, 0  ),  # 8
    (0,   0,   128),  # 9
    (128, 128, 0 ),   # 10
    (128, 0,   128),  # 11
    (0,   128, 128),  # 12
    (192, 192, 192),  # 13
    (64,  64,  64 ),  # 14
    (192, 0,   64 ),  # 15
    (64,  0,   192)   # 16
]

def id_to_rgb_mask(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cid, color in enumerate(PALETTE):
        rgb[mask == cid] = color
    return rgb

# Metrics
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        (n * label[k] + pred[k]).cpu().numpy(), minlength=n ** 2
    ).reshape(n, n)

def compute_miou(hist):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    return np.nanmean(iu), iu

def compute_dice(hist):
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0) + 1e-6)
    return np.nanmean(dice), dice

# Eval 
@torch.no_grad()
def evaluate(dataloader, model, device, save_root, mode="val"):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if mode == "test":
            imgs = batch.to(device, non_blocking=True)
            masks = None
        else:
            imgs, masks = batch
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

        logits = model(imgs)                       # (B,C,H,W)
        preds  = logits.argmax(1)                  # (B,H,W) channel 별로 max값 

        # visualize and save results
        for b in range(imgs.shape[0]):
            ct   = (imgs[b, 0].cpu().numpy() * 255).astype(np.uint8)
            pred = preds[b].cpu().numpy()
            pred_rgb = id_to_rgb_mask(pred)

            if masks is not None:
                gt   = masks[b].cpu().numpy()
                gt_rgb = id_to_rgb_mask(gt)
                mosaic = np.concatenate([ct[..., None]] * 3, axis=2)         # gray→RGB
                vis   = np.concatenate([mosaic, gt_rgb, pred_rgb], axis=1)   # H x 3W x 3
            else:
                mosaic = np.concatenate([ct[..., None]] * 3, axis=2)
                vis   = np.concatenate([mosaic, pred_rgb], axis=1)           # H x 2W x 3

            case_dir = save_root / f"sample_{idx*dataloader.batch_size+b:06d}"
            case_dir.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(vis).save(case_dir.with_suffix(".png"))

        # Metrics 
        if masks is not None:
            hist += fast_hist(preds.view(-1), masks.view(-1), NUM_CLASSES)

    # test dataset 경우에는 label이 없어 metrics 계산할 수 없음
    if mode != "test":
        miou, per_class_iou  = compute_miou(hist)
        mdice, per_class_dice = compute_dice(hist)

        # class별로 정확도 나타내는 json파일 생성
        metrics = {
            "mIoU": float(miou),
            "mDice": float(mdice),
            "IoU_per_class": [float(x) for x in per_class_iou],
            "Dice_per_class": [float(x) for x in per_class_dice],
        }
        (save_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("==> mIoU {:.3f} | mDice {:.3f}".format(miou, mdice))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="WORD-slices 변환본")
    parser.add_argument("--ckpt"     , required=True, help="학습된 모델 .pth")
    parser.add_argument("--save_dir" , default="./eval_vis")
    parser.add_argument("--batch"    , type=int, default=8)
    parser.add_argument("--workers"  , type=int, default=4)
    parser.add_argument("--mode"     , choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    ct_tf = transforms.Lambda(lambda t: (t - 0.5) / 0.5)
    ds = WORDdataset(args.data_root, mode=args.mode, ct_transform=ct_tf)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=NUM_CLASSES
    )
    model = model.to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    # Run
    save_root = Path(args.save_dir);  save_root.mkdir(parents=True, exist_ok=True)
    evaluate(loader, model, device, save_root, mode=args.mode)


if __name__ == "__main__":
    main()