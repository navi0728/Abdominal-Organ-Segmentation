from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch

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

_color2id = {}

for i, c in enumerate(PALETTE):
    _color2id[c] = i

CLUT = np.zeros((256, 256, 256), dtype=np.uint8) # Color Lookup Table 

for rgb, cls in _color2id.items():
    CLUT[rgb] = cls


def _rgb_mask_to_id(mask_rgb): # 기존의 3채널이 1채널로 변경됨
    return CLUT[
        mask_rgb[:, :, 0],
        mask_rgb[:, :, 1],
        mask_rgb[:, :, 2]
    ]


class WORDdataset(Dataset):
    def __init__(self, root, mode= "train", ct_transform= None, id_transform= None):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.ct_tf = ct_transform
        self.id_tf = id_transform

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
        # print("ct_img shape:", ct_img.size) # (512, 512)

        ct_arr = np.array(ct_img, dtype=np.float32)
        # print("ct_arr shape:", ct_arr.shape) # (512, 512)

        if ct_img.mode == "I;16":
            ct_arr = ct_arr / 65535.0
        ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0)
        # print("ct_tensor shape:", ct_tensor.shape) # torch.Size([1, 512, 512])

        if self.ct_tf:
            ct_tensor = self.ct_tf(ct_tensor)

        if mask_path is None:
            return ct_tensor

        mask_rgb = Image.open(mask_path)
        
        # print("mask_rgb shape:", mask_rgb.shape) # (512, 512, 3)

        mask_id  = torch.from_numpy(_rgb_mask_to_id(mask_rgb))
        # print("mask_id shape:", mask_id.shape)  # torch.Size([512, 512])

        if self.id_tf:
            mask_id = self.id_tf(mask_id)

        return ct_tensor, mask_id # torch.Size([1, 1, 512, 512]) # torch.Size([1, 512, 512])


if __name__ == "__main__":
    from torchvision import transforms

    ct_tf = transforms.Compose([
        transforms.Lambda(lambda t: (t - 0.5) / 0.5),   # transforms.Lambda(lambda x: 사용자정의함수(x))
        # 추가로 필요한 부분 전처리 (넣고 안 넣고 학습 후 비교)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])


    train_ds = WORDdataset(root="/mnt2/dataset/WORD/minju/WORD-slices", mode="train", ct_transform=ct_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True)

    # for i, ct in enumerate(train_loader):
    #     print(f"Batch {i}: CT shape: {ct.shape}")

    print(f"Train dataset: {len(train_ds)}") # Train dataset: 20115 -> 기존의 100에서 20115로 슬라이싱