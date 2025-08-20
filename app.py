import cv2
from ultralytics import YOLO
import numpy as np

class CoinRecognizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Todo ハードコーディングしているので外部に書く
        self.coin_values = {0: 1, 1: 100}  

    def recognize(self, img):
        results = self.model.predict(source=img, imgsz=640, conf=0.7, verbose=False)
        total_amount = 0
        total_boxes = 0
        output_img = img.copy()
        annotations = []  

        img_h, img_w = img.shape[:2]

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
                

                annotations.append(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                total_boxes += 1

                # --- 金額加算 --- 
                coin_value = int(self.coin_values.get(int(cls_id), 0))
                total_amount += coin_value

                # --- 描画 --- 
                label_text = f"#{idx}_{coin_value}yen"
                cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(output_img, label_text, (int(x1), max(int(y1), 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        return output_img, total_amount, annotations,img
