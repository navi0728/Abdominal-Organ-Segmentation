from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms.functional as F
import random

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


_color2id = {c: i for i, c in enumerate(PALETTE)}
CLUT = np.zeros((256, 256, 256), dtype=np.uint8)
for rgb, cid in _color2id.items():
    CLUT[rgb] = cid


def _rgb_mask_to_id(mask_rgb):
    return CLUT[
        mask_rgb[:, :, 0],
        mask_rgb[:, :, 1],
        mask_rgb[:, :, 2]
    ]

# 추가된 부분
class RandomCropResizeFlipRotate:
    def __init__(
        self,
        out_size = (512, 512),
        min_crop = 384,
        max_crop = 512,
        p_flip = 0.5,
        p_rotate = 0.5,
    ):
        self.out_h, self.out_w = out_size
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.p_flip   = p_flip
        self.p_rotate = p_rotate

    def _get_crop(self, h: int, w: int):
        th = random.randint(self.min_crop, min(self.max_crop, h))
        tw = random.randint(self.min_crop, min(self.max_crop, w))
        i  = random.randint(0, h - th)
        j  = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask = None):
        # img: (1,H,W) float
        _, h, w = img.shape
        i, j, th, tw = self._get_crop(h, w)

        img  = img[..., i:i+th, j:j+tw]                    # crop
        if mask is not None:
            mask = mask[i:i+th, j:j+tw]

        if random.random() < self.p_flip:
            img  = F.hflip(img)
            if mask is not None: 
                mask = F.hflip(mask)

        if random.random() < self.p_rotate:
            k = random.randint(1, 3)                       # 90/180/270
            img  = torch.rot90(img , k, [1, 2])
            if mask is not None:
                mask = torch.rot90(mask, k, [0, 1])

        img  = F.resize(img , (self.out_h, self.out_w), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        if mask is not None:
            mask = mask.unsqueeze(0).float()               # (1,H,W) float
            mask = F.resize(mask, (self.out_h, self.out_w), interpolation=F.InterpolationMode.NEAREST)
            mask = mask.squeeze(0).long()                  # back to (H,W) long

        return (img, mask) if mask is not None else img


class WORDdataset(Dataset):
    def __init__(self, root, mode= "train", ct_transform= None, id_transform= None):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.ct_tf = ct_transform
        self.id_tf = id_transform

        if mode == "train":
            self.joint_tf = RandomCropResizeFlipRotate()
        else:
            self.joint_tf = None

        self.root = self.root / mode

        self.samples = []
        cases = sorted(list(self.root.glob("*")))
        loop = tqdm(cases, desc="Indexing cases")
        for cdir in loop:
            ct_files = sorted((cdir / "ct").glob("*"))
            if (cdir / "mask").exists() and mode != "test":
                ms_files = sorted((cdir / "mask").glob("*"))
                assert len(ct_files) == len(ms_files), f"slice mismatch @ {cdir}"
                self.samples.extend(zip(ct_files, ms_files))
            else:
                self.samples.extend([(p, None) for p in ct_files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, mask_path = self.samples[idx]

        ct_img = Image.open(ct_path)
        ct_arr = np.array(ct_img, dtype=np.float32)
        if ct_img.mode == "I;16":
            ct_arr = ct_arr / 65535.0
        ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0)

        if self.ct_tf:
            ct_tensor = self.ct_tf(ct_tensor)

        if mask_path is None:
            return ct_tensor

        mask_rgb = Image.open(mask_path)
        mask_rgb = np.array(mask_rgb, dtype=np.uint8)
        mask_id  = torch.from_numpy(_rgb_mask_to_id(mask_rgb))

        if self.id_tf:
            mask_id = self.id_tf(mask_id)

        if self.joint_tf:
            ct_tensor, mask_id = self.joint_tf(ct_tensor, mask_id)

        return ct_tensor, mask_id


if __name__ == "__main__":
    from torchvision import transforms

    ct_tf = transforms.Compose([
        transforms.Lambda(lambda t: (t - 0.5) / 0.5)
    ])

    train_ds = WORDdataset(root="/mnt2/dataset/WORD/minju/WORD-slices", mode="train", ct_transform=ct_tf)


    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True)


    print(f"Train dataset: {len(train_ds)}")