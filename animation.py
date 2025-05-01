import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import SimpleITK as sitk


nii_img_path = "/mnt2/dataset/WORD/imagesTr/word_0002.nii.gz"

## 데이터 load 코드
# 지정된 경로에서 NIfTI 파일을 메모리에 로드
t_nii = nib.load(nii_img_path)

t_header = t_nii.header
print("Header 정보:", t_header) # 헤더 정보 확인

# 영상의 실제 픽셀 값을 numpy 배열로 반환
t_nii_data = t_nii.get_fdata()

print("Data shape:", t_nii_data.shape) # (512, 512, 241) -> 학습을 위한 데이터 shape인 (241, 512, 512)로 변경 필요


## 데이터 gif로 변환 코드
# 슬라이스 수 설정 (X, Y, Z축 기준)
n_slices = t_nii_data.shape[2]  # 전체 Z 슬라이스 수(축 변경 가능)
slice_range = range(n_slices)   # 또는 range(0, n) 등 원하는 범위 지정하여 slicing 가능

fig = plt.figure()

def update(i):
    # plt.clf()  
    slice_data = t_nii_data[:, :, i]
    plt.imshow(slice_data.T, cmap='gray')
    plt.title(f"Slice {i+1} / {n_slices}")
    plt.axis('off')

# 애니메이션 생성
ani = animation.FuncAnimation(fig, update, frames=slice_range, interval=100)

# 생성된 gif 파일 저장
ani.save("output_slices.gif", writer='pillow', fps=10)

print("GIF 저장 완료: output_slices.gif")


