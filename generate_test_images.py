"""
生成测试图片（用于快速验证）
如果已有真实图片，可以跳过此步骤
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))


def create_test_image(image_path, caption, size=(224, 224)):
    """创建带有文字标签的测试图片"""
    # 创建随机彩色背景
    arr = np.random.randint(50, 200, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(arr)

    # 添加文字标签
    draw = ImageDraw.Draw(image)

    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

    # 添加文字（白色背景+黑色文字）
    text = caption[:15]  # 截断文字
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # 文字背景
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(255, 255, 255)
    )
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    image.save(image_path)
    print(f"已生成: {image_path}")


def main():
    project_root = get_project_root()
    images_dir = os.path.join(project_root, 'data', 'images')
    os.makedirs(images_dir, exist_ok=True)

    captions = [
        "一只橘猫在沙发上睡觉",
        "日落时分的城市天际线",
        "程序员在深夜写代码",
        "一碗热气腾腾的拉面",
        "雪山脚下的湖泊",
        "樱花盛开的公园小径",
        "繁忙的十字路口",
        "一杯冒着热气的咖啡",
        "雨后的彩虹",
        "古老的图书馆书架",
    ]

    print("生成测试图片...")
    for i, caption in enumerate(captions, 1):
        image_path = os.path.join(images_dir, f"{i:03d}.jpg")
        create_test_image(image_path, caption)

    print(f"\n已生成 {len(captions)} 张测试图片到: {images_dir}")
    print("提示: 建议替换为真实图片以获得更好的训练效果")


if __name__ == '__main__':
    main()

