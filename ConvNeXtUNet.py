import torch, timm
import torch.nn as nn
import torch.nn.functional as F

from urllib.parse import urlparse


class ConvBnAct(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

# 기존 UNet의 encoder부분 유지
class UpBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1, self.conv2 = ConvBnAct(in_c, mid_c), ConvBnAct(mid_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, (0, skip.size(-1)-x.size(-1), 0, skip.size(-2)-x.size(-2)))
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# 기존 UNet의 encoder부분을 ConvNeXt로 변경 -> huggingface timm 사용 및 imagenet dataset으로 사전 학습한 weights 사용(사실 ct data와는 관련성 낮음)
class ConvNeXtUNet(nn.Module):
    def __init__(self, encoder_name = "convnext_tiny.in22k", num_classes = 1, encoder_pretrained = True): 
        super().__init__()
        # header를 제외한 backbone 불러오기
        # 기존의 classification용이 아닌 feature extraction용으로 encoder를 불러옴
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True)

        if encoder_pretrained:
            self._load_encoder_weights(encoder_pretrained)

        C1, C2, C3, C4 = self.encoder.feature_info.channels()

        self.bottleneck = nn.Sequential(
            ConvBnAct(C4, C4), ConvBnAct(C4, C4)
        )

        self.up4 = UpBlock(C4 + C3, C3, C3)
        self.up3 = UpBlock(C3 + C2, C2, C2)
        self.up2 = UpBlock(C2 + C1, C1, C1)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBnAct(C1, C1 // 2),
            nn.Conv2d(C1 // 2, num_classes, 1)
        )

    # pretrained encoder의 weight를 불러와 모델에 넣기
    def _load_encoder_weights(self, src):
        if isinstance(src, bool):
            self.encoder.load_state_dict(timm.create_model(self.encoder.default_cfg['architecture'], pretrained=True, features_only=True).state_dict(), strict=False)
        else:
            ckpt = torch.hub.load_state_dict_from_url(src, map_location="cpu") \
                   if urlparse(src).scheme in {"http", "https", "hf"} \
                   else torch.load(src, map_location="cpu")
            ckpt = {k.replace("model.", "").replace("backbone.", ""): v
                    for k, v in ckpt.items()}
            missing, unexpected = self.encoder.load_state_dict(ckpt, strict=False)
            print(f"[Encoder] missing {len(missing)}, unexpected {len(unexpected)} keys")

    def forward(self, x):
        # ConvNeXt 통과
        c1, c2, c3, c4 = self.encoder(x)
        # 기존 UNet의 bottleneck과 decoder 부분 통과
        x = self.bottleneck(c4)
        x = self.up4(x, c3)
        x = self.up3(x, c2)
        x = self.up2(x, c1)
        x = self.head(x)
        return x
    
    ## 학습이 완료된 모델의 파라미터(checkpoint)를 로드해, 모델 객체를 복원 하는 메소드
    # @classmethod
    # def from_checkpoint(cls, ckpt_path: str, **init_kwargs):
    #     model = cls(**init_kwargs)
    #     sd = torch.load(ckpt_path, map_location="cpu")
    #     if "state_dict" in sd:  sd = sd["state_dict"]
    #     sd = {k.replace("model.", ""): v for k, v in sd.items()}
    #     model.load_state_dict(sd, strict=True)
    #     return model


if __name__ == "__main__":
    m1 = ConvNeXtUNet("convnext_tiny.fb_in22k_ft_in1k", num_classes=2, encoder_pretrained=True)
    x = torch.randn(1, 3, 256, 256)
    print(m1(x).shape) # (B, num_classes, H, W)