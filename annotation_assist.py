
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# --- 設定 ---
ROOT_DIR = Path(__file__).parent
IMG_PATH = ROOT_DIR / "datasets/images/val/1000023526.jpg"

#IMG_PATH = ROOT_DIR / "datasets/images/train/1000023525.jpg"

MODEL_WEIGHTS = ROOT_DIR / "finetuning_result/train/weights/best.pt"
FORCE_CLASS_ID = None         # Noneなら推論結果のクラスを使用、整数なら強制

MODEL_WEIGHTS = "datasets/yolov8n.pt" 
#MODEL_WEIGHTS = "datasets/yolov8x.pt" 

# --- 画像読み込み ---
img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")
img_h, img_w = img.shape[:2]

# --- モデル読み込み＆推論 ---
model = YOLO(MODEL_WEIGHTS)
results = model.predict(source=str(IMG_PATH), imgsz=640, conf=0.015, verbose=False)

# --- アノテーション作成と描画 ---
lines = []
total_boxes = 0
annotated_img = img.copy()

for res in results:
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        continue

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)

    for idx, ((x1, y1, x2, y2), cls_id, conf) in enumerate(zip(xyxy, cls_ids, confs), start=1):
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        class_id = int(cls_id) if FORCE_CLASS_ID is None else int(FORCE_CLASS_ID)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        total_boxes += 1

        # --- 描画 ---
        label_text = f"#{idx} cls:{class_id} conf:{conf:.2f}"
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_img, label_text, (int(x1), max(int(y1)+100, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

# --- アノテーションtxt保存 ---
txt_path = IMG_PATH.with_suffix('.txt')
txt_path.write_text("\n".join(lines), encoding='utf-8')

# --- 描画結果保存 ---
annotated_path = IMG_PATH.with_name(IMG_PATH.stem + '_annotated.jpg')
cv2.imwrite(str(annotated_path), annotated_img)

print(f"YOLO annotation saved to: {txt_path}  (boxes: {total_boxes})")
print(f"Annotated image saved to: {annotated_path}")
print("アノテーション行番号 → 画像の #番号 と対応")
