import os
from pathlib import Path
from glob import glob
from tqdm import tqdm

import numpy as np
import nibabel as nib
from PIL import Image

root_dir = "/mnt2/dataset/WORD"
out_dir = "/mnt2/dataset/WORD/minju/WORD-slices/test"
mode = "test"
save_tiff = True
ct_subdir = "ct"
mask_subdir= "mask"

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

def _normalize_to_uint16(arr):
    arr = arr.astype(np.float32) # 1. float32로 변환
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) # 2. 0~1 정규화
    return (arr * 65535).round().astype(np.uint16) # 3. 0~65535로 스케일링 후 uint16 변환

def _mask_to_rgb(mask2d):
    h, w = mask2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(PALETTE): # ex) color: (0, 0, 0)
        # class 별로 해당 palette에 해당하는 픽셀을 RGB로 변환(mask 생성)
        rgb[mask2d == cls] = color
    return rgb

## img,mask 경로 불러오기
if mode == "train":
    img_paths  = sorted(glob(f"{root_dir}/imagesTr/*.nii.gz"))
    mask_paths = sorted(glob(f"{root_dir}/labelsTr/*.nii.gz"))

elif mode == "val":
    img_paths  = sorted(glob(f"{root_dir}/imagesVal/*.nii.gz"))
    mask_paths = sorted(glob(f"{root_dir}/labelsVal/*.nii.gz"))

else:
    img_paths, mask_paths = sorted(glob(f"{root_dir}/imagesTs/*.nii.gz")), None


for i, img_path in enumerate(tqdm(img_paths, desc="Saving slices")):

    ## 생성된 slice 이미지를 저장할 디렉토리 생성 
    # vol_name = Path(img_path).stem.replace(".nii", "")
    vol_name = os.path.basename(img_path).replace(".nii.gz", "")

    out_vol_dir = os.path.join(out_dir, vol_name)
    out_ct_dir = os.path.join(out_vol_dir, ct_subdir) # ct_subdir = "ct"
    out_mask_dir = os.path.join(out_vol_dir, mask_subdir) # mask_subdir= "mask"
    # out_vol_dir = Path(out_dir) / vol_name
    # out_ct_dir   = out_vol_dir / ct_subdir
    # out_mask_dir = out_vol_dir / mask_subdir
    
    os.makedirs(out_ct_dir, exist_ok=True)
    # out_ct_dir.mkdir(parents=True, exist_ok=True)
    if mask_paths: 
        os.makedirs(out_mask_dir, exist_ok=True)
        # out_mask_dir.mkdir(parents=True, exist_ok=True)

    # 지정된 경로에서 NIfTI 파일을 메모리에 로드
    img_vol = nib.load(img_path)

    # 영상의 실제 픽셀 값을 numpy 배열로 반환
    img_vol = img_vol.get_fdata()

    # 변환된 numpy 배열을 (D, H, W)로 변환
    img_vol = np.transpose(img_vol, (2, 0, 1))

    if mask_paths:
        mask_vol = nib.load(mask_paths[i])
        mask_vol = mask_vol.get_fdata().astype(np.uint8)
        mask_vol = np.transpose(mask_vol, (2, 0, 1))

    for z, slice2d in enumerate(img_vol):
        if save_tiff:

            # 생성된 ct_arr: 정규화되고 uint16 형식으로 변환된 2D NumPy 배열 (slice 이미지)
            ct_arr = _normalize_to_uint16(slice2d) 
            # ct_arr를 16비트 정수 이미지로 PIL 이미지 생성
            img_pil = Image.fromarray(ct_arr, mode="I;16")

            ext = "tiff"

        else:
            ct_arr = _normalize_to_uint16(slice2d) >> 8
            img_pil = Image.fromarray(ct_arr.astype(np.uint8), mode="L")

            ext = "png"

        # norm과 Image.fromarray를 통해 생성된 PIL이미지 img_pil를 파일 형태로 저장
        # save() 메서드: 이미지 형식에 맞는 파일 확장자와 함께 파일 경로를 지정하여 이미지를 디스크에 저장하는 역할
        img_pil.save(Path(out_ct_dir) / f"{z:04d}.{ext}", compression="tiff_deflate" if save_tiff else None)

        if mask_paths:
            mask_rgb = _mask_to_rgb(mask_vol[z])
            # mask_rgb를 16비트 정수 이미지로 PIL 이미지 생성(ct_arr와 동일한 과정)
            mask_rgb = Image.fromarray(mask_rgb, mode="RGB")

            # norm과 Image.fromarray를 통해 생성된 PIL이미지 mask_rgb를 파일 형태로 저장
            # save() 메서드: 이미지 형식에 맞는 파일 확장자와 함께 파일 경로를 지정하여 이미지를 디스크에 저장하는 역할
            mask_rgb.save(Path(out_mask_dir) / f"{z:04d}.png")