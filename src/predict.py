"""
单图推理模块
============
加载训练好的随机森林模型，对输入车辆图像进行颜色识别。

设计要点（面向边缘侧部署）：
  - 推理流程与训练完全一致（预处理 → HSV 特征提取 → RF 分类）
  - 不依赖 GPU，纯 CPU 执行，单张 ≤ 15ms
  - 对外暴露 predict_single() 函数，方便集成到上层感知系统

运行方式：
    python src/predict.py --image path/to/car.jpg
    python src/predict.py --image path/to/car.jpg --show
"""

import sys
import time
import argparse
import joblib
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess
from feature_extraction import extract_hsv_histogram
from utils import label_id_to_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "vcr_random_forest.pkl"


def load_model():
    """加载训练好的随机森林模型。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {MODEL_PATH}\n"
            f"请先运行 python src/train.py 进行训练。"
        )
    model = joblib.load(MODEL_PATH)
    return model


def predict_single(bgr_img: np.ndarray, model=None) -> dict:
    """
    对单张车辆图像进行颜色识别。

    完整推理流程：
        原始图像 → 预处理 → HSV 特征提取 → 随机森林推理 → 结果

    Args:
        bgr_img: 原始 BGR 图像（由 cv2.imread 读取）
        model: 已加载的模型对象，None 时自动加载
    Returns:
        dict 包含：
          - label_id (int): 预测类别编号
          - label_zh (str): 中文颜色名称
          - confidence (float): 置信度（随机森林投票比例）
          - time_ms (float): 推理耗时（毫秒）
    """
    if model is None:
        model = load_model()

    t_start = time.perf_counter()

    # Step 1: 图像预处理
    img_processed = preprocess(bgr_img)

    # Step 2: 特征提取（60 维 HSV 直方图）
    feat = extract_hsv_histogram(img_processed)
    feat_2d = feat.reshape(1, -1)  # sklearn 需要 2D 输入

    # Step 3: 随机森林推理
    # predict_proba 返回各类别的投票比例（即置信度）
    proba = model.predict_proba(feat_2d)[0]
    label_id = int(np.argmax(proba))
    confidence = float(proba[label_id])

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return {
        "label_id":   label_id,
        "label_zh":   label_id_to_name(label_id, lang="zh"),
        "confidence": confidence,
        "time_ms":    elapsed_ms,
    }


def visualize_result(bgr_img: np.ndarray, result: dict) -> np.ndarray:
    """
    在图像上叠加识别结果文字，方便直观查看。

    Args:
        bgr_img: 原始 BGR 图像
        result: predict_single() 返回的结果字典
    Returns:
        带标注文字的 BGR 图像
    """
    vis = bgr_img.copy()
    h, w = vis.shape[:2]

    # 绘制半透明背景矩形
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # 绘制文字（OpenCV 不支持中文，此处用英文标注；生产环境可用 PIL）
    label_en = result["label_zh"]
    text = f"Color: {label_en}  Conf: {result['confidence']:.1%}  {result['time_ms']:.1f}ms"
    cv2.putText(
        vis, text,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description="车辆颜色识别推理")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--show", action="store_true", help="显示结果图像")
    args = parser.parse_args()

    # 加载图像
    img = cv2.imread(args.image)
    if img is None:
        print(f" 无法读取图像: {args.image}")
        sys.exit(1)

    # 加载模型并推理
    model = load_model()
    result = predict_single(img, model=model)

    # 输出结果
    print("\n 识别结果：")
    print(f"  颜色类别: {result['label_zh']}")
    print(f"  置信度:   {result['confidence']:.2%}")
    print(f"  推理耗时: {result['time_ms']:.2f} ms")

    if args.show:
        vis = visualize_result(img, result)
        cv2.imshow("Vehicle Color Recognition", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
