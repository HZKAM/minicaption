"""
图文配对数据集加载器
"""
import json
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageCaptionDataset(Dataset):
    """图文配对数据集"""

    def __init__(self, jsonl_file, transform=None):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data = []
    
        if not os.path.isabs(jsonl_file):
            jsonl_file = os.path.join(self.project_root, jsonl_file)
    
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)
    
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        caption = item['caption']
    
        # ===== 关键：这里必须拼接项目根目录 =====
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.project_root, image_path)
        # ======================================
    
        image = Image.open(image_path).convert('RGB')
    
        # if self.transform:
        #    image = self.transform(image)
    
        return {
            'image': image,
            'caption': caption
        }
