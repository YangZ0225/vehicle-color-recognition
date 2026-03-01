"""
模型训练与评估脚本
==================
执行流程：
  1. 加载数据集
  2. 图像预处理（CLAHE + 双边滤波）
  3. HSV 特征提取
  4. 对比 KNN / SVM / 随机森林（RandomForest）
  5. 选最优模型，保存到 models/ 目录
  6. 生成混淆矩阵、分类报告、特征重要性图

运行方式：
    python src/train.py
"""

import os
import sys
import time
import joblib
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")   # 无头环境下使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Microsoft YaHei'  # Windows 微软雅黑
rcParams['axes.unicode_minus'] = False 

import matplotlib.font_manager as fm
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# ── 将 src 目录加入路径 ──
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess
from feature_extraction import extract_hsv_histogram, get_feature_names
from utils import (
    load_dataset, label_id_to_name, ensure_dir,
    print_class_distribution, LABEL_MAP, ID_TO_ZH
)

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = ensure_dir(PROJECT_ROOT / "models")
RESULTS_DIR  = ensure_dir(PROJECT_ROOT / "results")

# 模型保存路径
MODEL_PATH   = MODEL_DIR / "vcr_random_forest.pkl"
SCALER_PATH  = MODEL_DIR / "label_classes.npy"


def adjust_gamma(bgr_img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma 校正模拟不同光照条件。
    gamma < 1 → 图像变暗（模拟夜间/逆光）
    gamma > 1 → 图像变亮（模拟强光/过曝）
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(bgr_img, table)

def build_feature_matrix(images: list, labels: list) -> tuple:
    """
    对所有图像执行预处理 + 特征提取，构建特征矩阵。

    Args:
        images: 原始 BGR 图像列表
        labels: 对应整数标签列表
    Returns:
        (X, y) — 特征矩阵 shape=(N, 60)，标签数组 shape=(N,)
    """
    print("\n🔧 正在执行预处理 + 特征提取...")
    X, y = [], []

    for img, label in tqdm(zip(images, labels), total=len(images), desc="提取特征"):
        try:
            # Step 1: 预处理（CLAHE + 双边滤波 + ROI 裁剪）
            img_processed = preprocess(img)
            # Step 2: 提取 HSV 非均匀量化直方图特征（60维）
            feat = extract_hsv_histogram(img_processed)
            # Step 3: 保存原始图片的特征
            X.append(feat)
            y.append(label)

            # ── 数据增强：对每张图额外生成亮度变暗/变亮的版本 ──
            # 模拟夜间路灯、逆光等场景，扩充训练数据多样性
            for gamma in [0.5, 1.8]:
                augmented = adjust_gamma(img, gamma)
                aug_processed = preprocess(augmented)
                aug_feat = extract_hsv_histogram(aug_processed)
                X.append(aug_feat)
                y.append(label)

            
        except Exception as e:
            print(f"\n  ⚠️  跳过一张图像: {e}")
            continue

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def build_feature_matrix_no_aug(images: list, labels: list) -> tuple:
    """
    测试集专用：只做预处理和特征提取，不做数据增强。
    保证测试集模拟真实部署场景，评估结果真实可信。
    """
    print("\n 正在处理测试集（无增强）...")
    X, y = [], []
    for img, label in tqdm(zip(images, labels), total=len(images), desc="处理测试集"):
        try:
            img_processed = preprocess(img)
            feat = extract_hsv_histogram(img_processed)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"\n   跳过一张图像: {e}")
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def compare_models(X_train, y_train, X_test, y_test) -> dict:
    """
    对比 KNN / SVM / 随机森林 三种分类器的性能。

    为什么最终选随机森林：
      - 随机森林 = 多棵决策树的集成，每棵树随机采样特征/样本（Bagging）
      - 推理时走 If-Else 决策路径，纯 CPU 运算，天然适合边缘侧设备
      - 内置特征重要性，方便后续剔除冗余维度

    Args:
        X_train, y_train: 训练集
        X_test, y_test: 测试集
    Returns:
        结果字典 {模型名: {accuracy, time_ms, model}}
    """
    # ── 代价敏感学习 ──
    # class_weight='balanced' 让 sklearn 自动根据各类样本数反比计算权重
    # 使粉色、紫色等稀有颜色的误分代价更高，迫使模型更认真学习这些类别
    models = {
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale",
                         class_weight="balanced", random_state=42),
        "随机森林":  RandomForestClassifier(
            n_estimators=200,       # 200 棵树，平衡精度与速度
            max_depth=None,         # 不限深度，让树充分生长
            min_samples_split=5,    # 每个节点至少 5 个样本才可分裂，防过拟合
            class_weight="balanced",# 代价敏感学习：稀有颜色赋予高权重
            n_jobs=-1,              # 多核并行训练
            random_state=42
        ),
    }

    results = {}
    print("\n📊 模型对比实验：")
    print(f"  {'模型':<15} {'准确率':>8} {'推理时延(ms/样本)':>18}")
    print("  " + "-" * 45)

    for name, model in models.items():
        # 训练
        model.fit(X_train, y_train)

        # 评估推理时延（测试集上的平均耗时）
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start) / len(X_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": acc,
            "time_ms": elapsed_ms,
            "model": model,
            "y_pred": y_pred,
        }
        print(f"  {name:<15} {acc:>7.2%} {elapsed_ms:>17.2f}")

    return results


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """绘制混淆矩阵并保存为图片。"""
    labels_order = sorted(ID_TO_ZH.keys())
    label_names = [ID_TO_ZH[i] for i in labels_order]

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax
    )
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title("车辆颜色识别 — 混淆矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   混淆矩阵已保存: {save_path}")


def plot_feature_importance(rf_model, save_path: Path):
    """
    绘制随机森林特征重要性（Mean Decrease Impurity）。

    特征重要性的意义：
        随机森林每次分裂节点时，选择能最大降低不纯度（Gini/Entropy）的特征。
        对所有树所有节点的该特征贡献取平均，即为重要性分数。
        重要性低的特征可以剔除，降低特征维度，加速推理。
    """
    importances = rf_model.feature_importances_
    feature_names = get_feature_names()

    # 按重要性降序排列，只显示 Top 20
    indices = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(20), top_scores[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("特征重要性（Mean Decrease Impurity）", fontsize=11)
    ax.set_title("Top 20 重要特征 — HSV 直方图各维度", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   特征重要性图已保存: {save_path}")


def save_classification_report(y_true, y_pred, save_path: Path):
    """保存各类别的 Precision / Recall / F1 报告。"""
    labels_order = sorted(ID_TO_ZH.keys())
    label_names = [ID_TO_ZH[i] for i in labels_order]

    report = classification_report(
        y_true, y_pred,
        labels=labels_order,
        target_names=label_names
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("车辆颜色识别 — 分类报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"   分类报告已保存: {save_path}")
    print("\n" + report)


def main():
    print("=" * 55)
    print("  车辆颜色识别系统 — 训练流程")
    print("=" * 55)

    # ── Step 1: 加载原始图像 ──
    images, labels = load_dataset(verbose=True)
    if len(images) == 0:
        print("\n 未找到任何图像！请检查 data/raw/ 目录下是否有图片。")
        sys.exit(1)
    print_class_distribution(labels)

    # ── Step 2: 先划分原始图像，再分别处理 ──
    # 关键：必须在增强之前就划分好，防止同一张图的原图和增强版
    # 分别出现在训练集和测试集里，导致评估结果虚高
    idx = list(range(len(images)))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=42
    )

    train_images = [images[i] for i in idx_train]
    train_labels = [labels[i] for i in idx_train]
    test_images  = [images[i] for i in idx_test]
    test_labels  = [labels[i] for i in idx_test]

    print(f"\n  原始划分 → 训练集: {len(train_images)} 张 | 测试集: {len(test_images)} 张")

    # ── Step 3: 训练集做增强（扩充3倍），测试集不做增强 ──
    X_train, y_train = build_feature_matrix(train_images, train_labels)
    X_test,  y_test  = build_feature_matrix_no_aug(test_images, test_labels)

    print(f"\n  增强后 → 训练集: {len(X_train)} 条 | 测试集: {len(X_test)} 条")

    # ── Step 4: 对比三种分类器 ──
    results = compare_models(X_train, y_train, X_test, y_test)

    # ── Step 5: 选择随机森林作为最终模型 ──
    best_model = results["随机森林"]["model"]
    best_pred  = results["随机森林"]["y_pred"]

    print(f"\n 最终模型准确率: {results['随机森林']['accuracy']:.2%}")

    # ── Step 6: 保存模型 ──
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n 模型已保存: {MODEL_PATH}")

    # ── Step 7: 生成评估报告 ──
    print("\n 生成评估报告...")
    plot_confusion_matrix(y_test, best_pred,
                          RESULTS_DIR / "confusion_matrix.png")
    plot_feature_importance(best_model,
                            RESULTS_DIR / "feature_importance.png")
    save_classification_report(y_test, best_pred,
                                RESULTS_DIR / "classification_report.txt")

    print("\n🎉 训练完成！所有结果保存在 results/ 目录。")


if __name__ == "__main__":
    main()
