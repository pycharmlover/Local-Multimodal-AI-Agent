# modules/image_encoder.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import CLIP_PATH

# =====================
# 设备选择
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 加载 CLIP 模型
# =====================
processor = CLIPProcessor.from_pretrained(CLIP_PATH)
model = CLIPModel.from_pretrained(
    CLIP_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()


# =====================
# 图像 → 向量（CLIP Image Encoder）
# =====================
@torch.no_grad()
def embed_image(image_input) -> np.ndarray:
    """
    输入: 图片路径 (str) 或 PIL Image 对象
    输出: 归一化后的 CLIP 图像向量 (np.ndarray)
    """
    # 支持传入路径字符串或 PIL Image 对象
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]


# =====================
# 文本 → 向量（CLIP Text Encoder，用于以文搜图）
# =====================
@torch.no_grad()
def embed_clip_text(text: str) -> np.ndarray:
    """
    输入: 文本描述
    输出: 归一化后的 CLIP 文本向量 (np.ndarray)
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]
