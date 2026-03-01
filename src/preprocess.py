"""
图像预处理模块
=============
解决高速监控场景下的图像质量问题：
  - 弱光/过曝：CLAHE 自适应对比度增强
  - 传感器噪声：双边滤波（保留边缘）
  - 感兴趣区域：车身 ROI 裁剪（去掉背景干扰）
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# CLAHE 参数说明：
#   clipLimit  — 对比度放大上限，太大会过增强噪声
#   tileGridSize — 局部均衡化的块大小，越小局部性越强
# ──────────────────────────────────────────────
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(bgr_img: np.ndarray) -> np.ndarray:
    """
    对 BGR 图像的亮度通道做 CLAHE（对比度受限自适应直方图均衡化）。

    原理：
        普通直方图均衡化是全局操作，容易过度拉伸导致噪声放大。
        CLAHE 将图像划分为小块（tile）分别均衡化，再通过插值拼接，
        同时用 clipLimit 截断过高的对比度放大倍数，兼顾局部与全局。

    Args:
        bgr_img: 原始 BGR 图像（uint8）
    Returns:
        亮度增强后的 BGR 图像
    """
    # 转到 LAB 色彩空间——L 通道独立表示亮度，不影响色彩
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 只对亮度通道做均衡化，色彩通道保持原样
    l_enhanced = _CLAHE.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_bilateral_filter(bgr_img: np.ndarray) -> np.ndarray:
    """
    双边滤波去噪，保留车身边缘细节。

    原理：
        高斯滤波只考虑空间距离，会模糊边缘。
        双边滤波同时考虑「空间距离」和「像素值差异」两个权重：
          - 距离近 且 像素值相近 → 权重高（平滑同质区域）
          - 距离近 但 像素值差异大 → 权重低（保留边缘不模糊）
        非常适合去除车漆表面的高频噪点，同时保留车身轮廓。

    Args:
        bgr_img: 输入 BGR 图像
    Returns:
        滤波后的 BGR 图像
    """
    # d=9: 滤波邻域直径
    # sigmaColor=75: 颜色空间标准差，越大表示颜色差异越大也会被平滑
    # sigmaSpace=75: 坐标空间标准差，越大表示更远的像素也参与计算
    return cv2.bilateralFilter(bgr_img, d=9, sigmaColor=75, sigmaSpace=75)


def crop_vehicle_roi(bgr_img: np.ndarray,
                     top_ratio: float = 0.1,
                     bottom_ratio: float = 0.85) -> np.ndarray:
    """
    裁剪车身感兴趣区域（ROI），去掉车牌区域和天空背景。

    实际卡口抓拍图像中，上方是背景/天空，下方是路面和车牌，
    车身主体通常集中在图像纵向 10%~85% 的区域。
    裁掉无关区域可以降低背景颜色对特征提取的干扰。

    Args:
        bgr_img: 输入 BGR 图像
        top_ratio: 从顶部裁去的比例
        bottom_ratio: 保留到底部的比例
    Returns:
        裁剪后的车身 ROI 图像
    """
    h = bgr_img.shape[0]
    top = int(h * top_ratio)
    bottom = int(h * bottom_ratio)
    return bgr_img[top:bottom, :]


def preprocess(bgr_img: np.ndarray,
               target_size: tuple = (128, 128)) -> np.ndarray:
    """
    完整预处理流水线入口。

    执行顺序：
        1. 尺寸统一（resize）
        2. CLAHE 亮度增强
        3. 双边滤波去噪
        4. 车身 ROI 裁剪
        5. 再次 resize 到统一大小（ROI 裁剪后尺寸变了）

    Args:
        bgr_img: 原始 BGR 图像
        target_size: 输出图像尺寸 (width, height)
    Returns:
        预处理完成的 BGR 图像，尺寸为 target_size
    """
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("输入图像为空，请检查图片路径是否正确")

    # 1. 先统一到较大尺寸，保留足够细节
    img = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)

    # 2. CLAHE 增强：解决弱光卡口图像对比度低的问题
    img = apply_clahe(img)

    # 3. 双边滤波：去除车漆反光和传感器噪点
    img = apply_bilateral_filter(img)

    # 4. 裁剪车身 ROI：去除无关背景
    img = crop_vehicle_roi(img)

    # 5. 统一到目标尺寸，供特征提取使用
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


if __name__ == "__main__":
    # 快速测试预处理效果
    import sys
    if len(sys.argv) > 1:
        raw = cv2.imread(sys.argv[1])
        if raw is None:
            print(f"无法读取图像: {sys.argv[1]}")
            sys.exit(1)
        processed = preprocess(raw)
        cv2.imshow("原始图像", raw)
        cv2.imshow("预处理结果", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("用法: python preprocess.py <图片路径>")
