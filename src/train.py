"""
图文配对模型训练脚本
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageCaptionDataset
from model import ImageTextModel


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def collate_fn(batch):
    """自定义 collate：保持 PIL Image 列表，文本保持列表"""
    images = [item['image'] for item in batch]
    texts = [item['caption'] for item in batch]
    return {
        'image': images,  # PIL Image 列表
        'caption': texts
    }

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # images = batch['image'].to(device)
        images = batch['image']
        texts = batch['caption']
        batch_size = len(images)

        # 前向传播
        logits = model(images, texts)

        # 对比学习标签：对角线为正样本
        labels = torch.arange(batch_size, device=device)

        # 计算损失（图片到文本 + 文本到图片）
        loss_i2t = criterion(logits, labels)
        loss_t2i = criterion(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # 配置
    project_root = get_project_root()
    data_file = os.path.join(project_root, 'data', 'captions.jsonl')
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 超参数
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-5
    image_size = 224

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 图像预处理
    # transform = transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=(0.48145466, 0.4578275, 0.40821073),
    #        std=(0.26862954, 0.26130258, 0.27577711)
    #    )
    #])
    dataset = ImageCaptionDataset(data_file)

    # 加载数据集
    print(f"加载数据集: {data_file}")
    # dataset = ImageCaptionDataset(data_file, transform=transform)
    dataset = ImageCaptionDataset(data_file)
    print(f"数据集大小: {len(dataset)}")

    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据文件路径")
        return

    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # 丢弃不完整的批次
        collate_fn=collate_fn
    )

    # 初始化模型
    print("加载 CLIP 模型...")
    model = ImageTextModel()
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print(f"\n开始训练 {num_epochs} 个 epoch...")
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}/{num_epochs} - 平均损失: {avg_loss:.4f}")

    # 保存模型
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"\n模型已保存到: {model_path}")
    print("训练完成！")


if __name__ == '__main__':
    main()

