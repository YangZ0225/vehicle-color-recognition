"""
工具函数模块
============
包含：数据加载、标签映射、路径管理等通用功能。
"""

import os
import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# 标签定义
# 将文件夹名映射为可读的中文标签和整数编码
# ──────────────────────────────────────────────
LABEL_MAP = {
    "black_blue_brown_purple": {"id": 0, "zh": "黑/蓝/棕/紫"},
    "green":                   {"id": 1, "zh": "绿色"},
    "orange_yellow":           {"id": 2, "zh": "橙/黄色"},
    "pink_red":                {"id": 3, "zh": "粉/红色"},
    "white_silver_gray":       {"id": 4, "zh": "白/银/灰"},
}

# 反向映射：id → 中文标签
ID_TO_ZH = {v["id"]: v["zh"] for v in LABEL_MAP.values()}
ID_TO_EN = {v["id"]: k for k, v in LABEL_MAP.items()}

# 支持的图像格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_data_root() -> Path:
    """
    自动定位 data/raw 目录（从项目根目录向下查找）。
    """
    # 从当前文件所在目录向上找到项目根
    current = Path(__file__).resolve().parent.parent
    data_root = current / "data" / "raw"
    return data_root


def load_dataset(data_root: str = None, verbose: bool = True) -> tuple:
    """
    从 data/raw 目录加载所有图像，返回图像列表和对应标签。

    目录结构要求：
        data/raw/
          ├── black_blue_brown_purple/  ← 黑蓝棕紫色车辆图片
          ├── green/
          ├── orange_yellow/
          ├── pink_red/
          └── white_silver_gray/

    Args:
        data_root: 数据根目录路径，None 时自动检测
        verbose: 是否打印加载进度
    Returns:
        (images, labels) — BGR 图像列表 和 整数标签列表
    """
    if data_root is None:
        data_root = get_data_root()
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"数据目录不存在: {data_root}\n"
            f"请将图像按颜色分类放入以下子目录：\n"
            + "\n".join(f"  {data_root / k}" for k in LABEL_MAP)
        )

    images, labels = [], []
    class_counts = {}

    for folder_name, meta in LABEL_MAP.items():
        folder_path = data_root / folder_name
        if not folder_path.exists():
            if verbose:
                print(f"!  目录不存在，跳过: {folder_path}")
            continue

        count = 0
        for file_path in sorted(folder_path.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            img = cv2.imread(str(file_path))
            if img is None:
                if verbose:
                    print(f"  !  无法读取: {file_path.name}")
                continue

            images.append(img)
            labels.append(meta["id"])
            count += 1

        class_counts[meta["zh"]] = count
        if verbose:
            print(f"   {meta['zh']:12s}: {count:4d} 张")

    if verbose:
        print(f"\n 共加载 {len(images)} 张图像，{len(LABEL_MAP)} 个类别")

    return images, labels


def label_id_to_name(label_id: int, lang: str = "zh") -> str:
    """
    将整数标签转换为可读名称。

    Args:
        label_id: 整数标签
        lang: "zh" 中文 / "en" 英文
    Returns:
        标签名称字符串
    """
    if lang == "zh":
        return ID_TO_ZH.get(label_id, f"未知({label_id})")
    return ID_TO_EN.get(label_id, f"unknown({label_id})")


def ensure_dir(path: str) -> Path:
    """创建目录（如果不存在）并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def print_class_distribution(labels: list):
    """打印类别分布统计，方便观察数据是否不平衡。"""
    from collections import Counter
    counter = Counter(labels)
    total = len(labels)
    print("\n 类别分布：")
    for label_id in sorted(counter.keys()):
        count = counter[label_id]
        ratio = count / total * 100
        bar = "█" * int(ratio / 2)
        print(f"  {label_id_to_name(label_id):12s} | {count:4d} 张 ({ratio:5.1f}%) {bar}")
