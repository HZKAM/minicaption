"""
图文配对模型推理脚本
"""
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

from model import ImageTextModel


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path, device):
    """加载训练好的模型"""
    model = ImageTextModel()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {model_path}")
    else:
        print("未找到训练好的模型，使用预训练权重")

    model = model.to(device)
    model.eval()
    return model


def predict(model, image_path, candidate_texts, device):
    """
    对单张图片进行图文匹配预测

    Args:
        model: 图文配对模型
        image_path: 图片路径
        candidate_texts: 候选文本列表
        device: 计算设备

    Returns:
        排序后的 (文本, 概率) 列表
    """
    # 图像预处理
    # transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=(0.48145466, 0.4578275, 0.40821073),
    #        std=(0.26862954, 0.26130258, 0.27577711)
    #    )
    # ])

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    # image_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        # logits = model(image_tensor, candidate_texts)
        logits = model(image, candidate_texts)
        probs = logits.softmax(dim=1)

    # 排序结果
    probs = probs[0].cpu().numpy()
    results = [(text, float(prob)) for text, prob in zip(candidate_texts, probs)]
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def main():
    # 配置
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'outputs', 'model.pt')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(model_path, device)

    # 测试图片
    test_images = [
        os.path.join(project_root, 'data', 'images', '001.jpg'),
        os.path.join(project_root, 'data', 'images', '002.jpg'),
    ]

    # 候选文本
    candidate_texts = [
        "一只橘猫在沙发上睡觉",
        "日落时分的城市天际线",
        "程序员在深夜写代码",
        "一碗热气腾腾的拉面",
        "雪山脚下的湖泊",
    ]

    print("\n" + "=" * 50)
    print("图文配对推理测试")
    print("=" * 50)

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"\n跳过不存在的图片: {image_path}")
            continue

        print(f"\n图片: {os.path.basename(image_path)}")
        print("-" * 40)

        results = predict(model, image_path, candidate_texts, device)

        for rank, (text, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 30)
            print(f"  {rank}. {prob:.4f} {bar} {text}")

    print("\n" + "=" * 50)
    print("推理完成！")


if __name__ == '__main__':
    main()

