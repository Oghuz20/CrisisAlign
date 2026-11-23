"""Text-only, image-only and late-fusion multimodal classifiers.

All models return a dict of logits:
    text   (B, 5)  humanitarian class of the caption
    image  (B, 5)  humanitarian class of the image
    agree  (B,)    1 = caption and image tell the same story (multimodal only)

Late fusion: each modality is encoded separately (DistilBERT, MobileNetV3),
projected to 256 dims, and the two vectors are concatenated. A missing
modality (has_text / has_image = 0, or modality dropout during training) is
zeroed and replaced by a learned "missing" vector, so the model knows it is absent.
"""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from transformers import AutoModel

from src.config import CLASSES

N_CLASSES = len(CLASSES)


class TextEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.dim = self.bert.config.hidden_size if hasattr(self.bert.config, "hidden_size") \
            else self.bert.config.dim

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features, self.pool = net.features, net.avgpool
        self.proj = net.classifier[:2]              # Linear(960 -> 1280) + Hardswish
        self.dim = 1280

    def forward(self, image):
        return self.proj(torch.flatten(self.pool(self.features(image)), 1))


def head(in_dim, out_dim, dropout=0.3):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


class TextOnly(nn.Module):
    def __init__(self, text_model, **_):
        super().__init__()
        self.text_enc = TextEncoder(text_model)
        self.text_head = head(self.text_enc.dim, N_CLASSES)

    def forward(self, b):
        return {"text": self.text_head(self.text_enc(b["input_ids"], b["attention_mask"]))}


class ImageOnly(nn.Module):
    def __init__(self, pretrained=True, **_):
        super().__init__()
        self.image_enc = ImageEncoder(pretrained)
        self.image_head = head(self.image_enc.dim, N_CLASSES)

    def forward(self, b):
        return {"image": self.image_head(self.image_enc(b["image"]))}


class LateFusion(nn.Module):
    def __init__(self, text_model, pretrained=True, proj_dim=256, modality_dropout=0.0):
        super().__init__()
        self.text_enc, self.image_enc = TextEncoder(text_model), ImageEncoder(pretrained)
        self.text_proj = nn.Sequential(nn.Linear(self.text_enc.dim, proj_dim), nn.GELU())
        self.image_proj = nn.Sequential(nn.Linear(self.image_enc.dim, proj_dim), nn.GELU())
        self.missing_text = nn.Parameter(torch.zeros(proj_dim))
        self.missing_image = nn.Parameter(torch.zeros(proj_dim))
        self.p = modality_dropout
        fused = 2 * proj_dim
        self.text_head = head(fused, N_CLASSES)
        self.image_head = head(fused, N_CLASSES)
        self.agree_head = head(fused, 1)

    def _mask(self, has, ref):
        has = has.to(ref.dtype)
        if self.training and self.p > 0:            # modality dropout: hide one input at random
            has = has * (torch.rand_like(has) >= self.p).to(ref.dtype)
        return has.unsqueeze(1)

    def forward(self, b):
        t = self.text_proj(self.text_enc(b["input_ids"], b["attention_mask"]))
        v = self.image_proj(self.image_enc(b["image"]))
        mt, mv = self._mask(b["has_text"], t), self._mask(b["has_image"], v)
        if self.training and self.p > 0:            # never hide both at once
            both = (mt + mv) == 0
            mv = torch.where(both, torch.ones_like(mv), mv)
        t = mt * t + (1 - mt) * self.missing_text
        v = mv * v + (1 - mv) * self.missing_image
        z = torch.cat([t, v], dim=1)
        return {"text": self.text_head(z), "image": self.image_head(z),
                "agree": self.agree_head(z).squeeze(1)}


def build_model(kind, text_model, pretrained=True, modality_dropout=0.0):
    if kind == "text":
        return TextOnly(text_model=text_model)
    if kind == "image":
        return ImageOnly(pretrained=pretrained)
    if kind == "multimodal":
        return LateFusion(text_model, pretrained, modality_dropout=modality_dropout)
    raise ValueError(kind)


def param_groups(model, lr_text, lr_image, lr_head, weight_decay):
    groups = {"text": [], "image": [], "head": []}
    for name, p in model.named_parameters():
        key = "text" if name.startswith("text_enc") else "image" if name.startswith("image_enc") else "head"
        groups[key].append(p)
    lrs = {"text": lr_text, "image": lr_image, "head": lr_head}
    return [{"params": ps, "lr": lrs[k], "weight_decay": weight_decay}
            for k, ps in groups.items() if ps]
