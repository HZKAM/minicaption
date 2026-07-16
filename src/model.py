"""
图文配对模型定义
基于 CLIP 的对比学习
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class ImageTextModel(nn.Module):
    """基于 CLIP 的图文配对模型"""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Args:
            model_name: HuggingFace 上的 CLIP 模型名称
        """
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images, texts):
        """
        前向传播

        Args:
            images: 图片张量 (batch_size, 3, 224, 224)
            texts: 文本列表 [str]

        Returns:
            logits_per_image: 图片到文本的相似度分数 (batch_size, batch_size)
        """
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 将数据移动到与模型相同的设备
        inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}

        outputs = self.clip(**inputs)

        # 返回图片到文本的 logits
        return outputs.logits_per_image

    def encode_image(self, images):
        """编码图片为向量"""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}
        return self.clip.get_image_features(**inputs)

    def encode_text(self, texts):
        """编码文本为向量"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}
        return self.clip.get_text_features(**inputs)

