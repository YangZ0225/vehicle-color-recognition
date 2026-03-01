"""
特征提取模块
============
核心思路：放弃 RGB，改用 HSV 色彩空间提取非均匀量化直方图。

为什么不用 RGB？
    RGB 三通道耦合了亮度信息。同一辆蓝色轿车，
    在强光下 RGB 值是 (100, 150, 220)，在阴影下可能是 (40, 60, 88)，
    数值差异极大，但颜色本质相同。这会导致分类器"认不出来"。

为什么用 HSV？
    HSV 将颜色解耦为：
      H（Hue/色调）    — 颜色的本质，如红/蓝/绿，不受亮度影响
      S（Saturation/饱和度） — 颜色的纯度
      V（Value/亮度）  — 像素的明暗程度
    通过重点建模 H 和 S，大幅抑制路灯强光、车身反光引起的特征漂移。

非均匀量化策略：
    - H 通道：细分（更多 bins）→ 捕捉颜色细节差异
    - S 通道：中等粒度
    - V 通道：粗分（少 bins）→ 亮度只是辅助信息，不需要太精细
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 非均匀量化 bins 设置
#   H 通道：0-180（OpenCV 中 Hue 范围是 0-179）→ 细分 36 bins
#   S 通道：0-255 → 中等 16 bins
#   V 通道：0-255 → 粗分 8 bins（亮度差异对颜色识别贡献小）
# 最终特征维度 = 36 + 16 + 8 = 60
# ──────────────────────────────────────────────
H_BINS = 36
S_BINS = 16
V_BINS = 8

FEATURE_DIM = H_BINS + S_BINS + V_BINS  # 60 维


def extract_hsv_histogram(bgr_img: np.ndarray,
                           mask: np.ndarray = None) -> np.ndarray:
    """
    从预处理后的 BGR 图像中提取 HSV 非均匀量化直方图特征。

    Args:
        bgr_img: 预处理后的 BGR 图像
        mask: 可选掩码（只统计 mask 区域内的像素，用于精准提取车身区域）
    Returns:
        shape=(FEATURE_DIM,) 的归一化特征向量（float32）
    """
    # 转换色彩空间：BGR → HSV
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # ── 分别对三个通道计算直方图 ──
    # cv2.calcHist 参数: [图像], [通道索引], 掩码, [bins数], [值域范围]

    # H 通道：色调，OpenCV 中范围 [0, 180)
    hist_h = cv2.calcHist([hsv], [0], mask, [H_BINS], [0, 180])

    # S 通道：饱和度，范围 [0, 256)
    hist_s = cv2.calcHist([hsv], [1], mask, [S_BINS], [0, 256])

    # V 通道：亮度，范围 [0, 256)
    hist_v = cv2.calcHist([hsv], [2], mask, [V_BINS], [0, 256])

    # ── 归一化：L1 归一化，使直方图和为 1 ──
    # 这样不同分辨率、不同 ROI 大小的图像可以直接比较
    def l1_normalize(hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        if total > 0:
            return (hist / total).flatten().astype(np.float32)
        return hist.flatten().astype(np.float32)

    feat_h = l1_normalize(hist_h)
    feat_s = l1_normalize(hist_s)
    feat_v = l1_normalize(hist_v)

    # ── 拼接为一个特征向量 ──
    feature_vector = np.concatenate([feat_h, feat_s, feat_v])

    return feature_vector


def extract_features_batch(images: list) -> np.ndarray:
    """
    批量提取特征，用于训练阶段。

    Args:
        images: 预处理后的 BGR 图像列表
    Returns:
        shape=(N, FEATURE_DIM) 的特征矩阵
    """
    features = []
    for img in images:
        feat = extract_hsv_histogram(img)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def get_feature_names() -> list:
    """
    返回每个特征维度的语义名称，用于特征重要性可视化。

    Returns:
        长度为 FEATURE_DIM 的字符串列表
    """
    names = []
    # H 通道特征名：每个 bin 对应的色调角度范围
    h_step = 180 / H_BINS
    for i in range(H_BINS):
        names.append(f"H_{int(i * h_step)}-{int((i+1) * h_step)}")

    # S 通道特征名
    s_step = 256 / S_BINS
    for i in range(S_BINS):
        names.append(f"S_{int(i * s_step)}-{int((i+1) * s_step)}")

    # V 通道特征名
    v_step = 256 / V_BINS
    for i in range(V_BINS):
        names.append(f"V_{int(i * v_step)}-{int((i+1) * v_step)}")

    return names


if __name__ == "__main__":
    # 快速验证特征提取
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"无法读取: {sys.argv[1]}")
            sys.exit(1)
        feat = extract_hsv_histogram(img)
        print(f"特征维度: {feat.shape}")
        print(f"特征值范围: [{feat.min():.4f}, {feat.max():.4f}]")
        print(f"特征向量（前10维）: {feat[:10]}")
    else:
        print(f"特征维度配置: H={H_BINS} + S={S_BINS} + V={V_BINS} = {FEATURE_DIM} 维")
