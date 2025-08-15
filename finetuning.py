import os
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # --- パス設定 (相対パス) ---
    # このスクリプトがあるディレクトリを基準とする
    ROOT_DIR = Path(__file__).parent
    MODEL_PATH = ROOT_DIR / "datasets/yolov8n.pt"
    DATA_YAML_PATH = ROOT_DIR / "datasets/data.yaml"
    PROJECT_PATH = ROOT_DIR / "finetuning_result"

    # --- モデル読み込み ---
    # 事前学習済みモデルが存在するかチェック
    if not MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))

    else:
        model = YOLO(MODEL_PATH)



    # --- 学習実行 ---
    model.train(
        data=str(DATA_YAML_PATH),
        lr0=0.001,
        epochs=100,
        batch=2,
        imgsz=640,
        mosaic=0.3,
        degrees=170.0,
        translate=0.2,
        perspective=0.001,
        scale=0.5,
        project=str(PROJECT_PATH),
        save=True,
        exist_ok=True,
        verbose=True
    )
