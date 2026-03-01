"""
单元测试
========
对预处理、特征提取、推理流程进行基本验证。
运行方式：
    python -m pytest tests/ -v
    或
    python tests/test_pipeline.py
"""

import sys
import numpy as np
from pathlib import Path

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import preprocess, apply_clahe, apply_bilateral_filter
from feature_extraction import (
    extract_hsv_histogram, FEATURE_DIM, H_BINS, S_BINS, V_BINS
)
from utils import label_id_to_name, LABEL_MAP


class TestPreprocess:
    """预处理模块测试"""

    def _make_dummy_image(self, h=200, w=200):
        """生成一张随机测试图像（模拟真实场景）"""
        # 模拟一辆蓝色汽车：中间区域偏蓝，整体加入一些噪声
        img = np.random.randint(30, 120, (h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(100, 200, (h, w))  # 蓝色通道偏高
        return img

    def test_clahe_output_shape(self):
        """CLAHE 不改变图像尺寸"""
        img = self._make_dummy_image()
        result = apply_clahe(img)
        assert result.shape == img.shape, "CLAHE 输出尺寸不匹配"
        print("✅ test_clahe_output_shape 通过")

    def test_bilateral_filter_output_shape(self):
        """双边滤波不改变图像尺寸"""
        img = self._make_dummy_image()
        result = apply_bilateral_filter(img)
        assert result.shape == img.shape, "双边滤波输出尺寸不匹配"
        print("✅ test_bilateral_filter_output_shape 通过")

    def test_preprocess_output_size(self):
        """预处理后输出尺寸为 (128, 128)"""
        img = self._make_dummy_image(300, 400)  # 非标准尺寸
        result = preprocess(img, target_size=(128, 128))
        assert result.shape == (128, 128, 3), f"输出尺寸错误: {result.shape}"
        print("✅ test_preprocess_output_size 通过")

    def test_preprocess_empty_input(self):
        """空图像应抛出 ValueError"""
        try:
            preprocess(np.array([]))
            assert False, "应该抛出异常"
        except ValueError:
            print("✅ test_preprocess_empty_input 通过")


class TestFeatureExtraction:
    """特征提取模块测试"""

    def _make_dummy_image(self, h=128, w=128):
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_feature_dimension(self):
        """特征向量维度应为 FEATURE_DIM（60）"""
        img = self._make_dummy_image()
        feat = extract_hsv_histogram(img)
        assert feat.shape == (FEATURE_DIM,), f"特征维度错误: {feat.shape}"
        print(f"✅ test_feature_dimension 通过 (dim={FEATURE_DIM})")

    def test_feature_normalized(self):
        """每个通道直方图独立归一化，值应在 [0, 1]"""
        img = self._make_dummy_image()
        feat = extract_hsv_histogram(img)
        assert feat.min() >= 0.0, "特征值不能为负"
        assert feat.max() <= 1.0, "特征值不能超过 1"
        print("✅ test_feature_normalized 通过")

    def test_feature_dtype(self):
        """特征类型应为 float32"""
        img = self._make_dummy_image()
        feat = extract_hsv_histogram(img)
        assert feat.dtype == np.float32, f"特征类型错误: {feat.dtype}"
        print("✅ test_feature_dtype 通过")

    def test_blue_vs_red_different(self):
        """蓝色图像和红色图像提取的特征应有显著差异"""
        blue_img = np.zeros((128, 128, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 200  # BGR: 高蓝色

        red_img = np.zeros((128, 128, 3), dtype=np.uint8)
        red_img[:, :, 2] = 200   # BGR: 高红色

        feat_blue = extract_hsv_histogram(blue_img)
        feat_red  = extract_hsv_histogram(red_img)

        # 余弦距离应该较大（两种颜色差异明显）
        cos_sim = np.dot(feat_blue, feat_red) / (
            np.linalg.norm(feat_blue) * np.linalg.norm(feat_red) + 1e-8
        )
        assert cos_sim < 0.9, f"蓝色和红色特征过于相似 (cos_sim={cos_sim:.3f})"
        print(f"✅ test_blue_vs_red_different 通过 (cos_sim={cos_sim:.3f})")


class TestUtils:
    """工具函数测试"""

    def test_label_mapping(self):
        """所有标签都能正确映射为中文名称"""
        for folder_name, meta in LABEL_MAP.items():
            name = label_id_to_name(meta["id"], lang="zh")
            assert name != f"未知({meta['id']})", f"标签映射失败: {meta['id']}"
        print("✅ test_label_mapping 通过")

    def test_invalid_label(self):
        """无效标签应返回 '未知' 而不是崩溃"""
        name = label_id_to_name(99, lang="zh")
        assert "未知" in name or "unknown" in name
        print("✅ test_invalid_label 通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 45)
    print("  车辆颜色识别系统 — 单元测试")
    print("=" * 45 + "\n")

    test_classes = [TestPreprocess, TestFeatureExtraction, TestUtils]
    total, passed = 0, 0

    for cls in test_classes:
        instance = cls()
        print(f"📋 {cls.__name__}")
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method in methods:
            total += 1
            try:
                getattr(instance, method)()
                passed += 1
            except Exception as e:
                print(f"  ❌ {method} 失败: {e}")
        print()

    print(f"{'=' * 45}")
    print(f"  测试结果: {passed}/{total} 通过")
    if passed == total:
        print("  🎉 所有测试通过！")
    print("=" * 45)


if __name__ == "__main__":
    run_all_tests()
