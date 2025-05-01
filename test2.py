import SimpleITK as sitk

# NIfTI 파일 경로
nii_path ="/mnt2/dataset/WORD/imagesTr/word_0002.nii.gz"  

# 이미지 읽기
image = sitk.ReadImage(nii_path)

# 이미지의 주요 정보 확인
spacing = image.GetSpacing()        # (x_spacing, y_spacing, z_spacing)
size = image.GetSize()              # (x_size, y_size, z_size)
origin = image.GetOrigin()          # 원점 좌표
direction = image.GetDirection()    # 방향 코사인 행렬

# 정보 출력
print(f"Pixel Spacing (x, y, z): {spacing}")
print(f"Image Size (x, y, z): {size}")
print(f"Origin: {origin}")
print(f"Direction (cosine matrix): {direction}")

# slice 두께는 일반적으로 z축 spacing을 의미함
print(f"Slice Thickness (z spacing): {spacing[2]}")



