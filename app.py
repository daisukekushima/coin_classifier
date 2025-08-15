import cv2
from ultralytics import YOLO

class CoinRecognizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # to do ハードコーディングしているので外部に書く
        self.coin_values = {0: 1, 1: 100}  

    def recognize(self, img):
        results = self.model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        total_amount = 0
        output_img = img.copy()
        annotations = []  #

        h, w = img.shape[:2]

        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                coin_value = self.coin_values.get(class_id, 0)
                total_amount += coin_value

                # YOLO形式に変換
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # 画像に描画
                label = f"{coin_value} yen (conf: {conf:.2f})"
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 合計金額を画像に描画
        #cv2.putText(output_img, f"Total: {total_amount} yen", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        return output_img, total_amount, annotations,img
