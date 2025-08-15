import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import os
from datetime import datetime
from pathlib import Path

from app import CoinRecognizer

# モデルとログ設定
model_path = r"finetuning_result/train/weights/best.pt"
recognizer = CoinRecognizer(model_path=model_path)
log_dir = "classifier_result"
os.makedirs(log_dir, exist_ok=True) 

# セッション初期化
st.session_state.setdefault("recognized", False)
st.session_state.setdefault("amount", None)
st.session_state.setdefault("result_img", None)

# 動画処理クラス
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_time = 0
        self.interval = 1.0
        self.result_img = None
        self.recognized = False
        self.amount = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = cv2.getTickCount() / cv2.getTickFrequency()
        output_img = img

        if self.recognized and self.result_img is not None:
            return av.VideoFrame.from_ndarray(self.result_img, format="bgr24")

        if now - self.last_time >= self.interval:
            self.last_time = now
            detected_img, total_amount, annotations, raw_img = recognizer.recognize(img)
            print(f"[recv] detected_img shape={detected_img.shape}, total_amount={total_amount}")

            if total_amount and total_amount > 0:
                self.result_img = detected_img
                self.amount = total_amount
                self.recognized = True
                self.raw_img = raw_img
                self.annotations = annotations
                output_img = detected_img
            else:
                output_img = detected_img

        return av.VideoFrame.from_ndarray(output_img, format="bgr24")


# タイトル
st.title("💰 硬貨識別ツール")

# 左右カラム配置
col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1920}, "height": {"ideal": 1080}},
            "audio": False
        }
    )

with col2:
    st.subheader("認識結果")

    if ctx.video_processor:
        processor = ctx.video_processor
        processor.amount = None
        processor.result_img = None
        processor.recognized = False

        # 認識が完了するまで待機
        while not processor.recognized:
            time.sleep(0.5)

        if processor.recognized:
            st.session_state["recognized"] = True
            st.session_state["amount"] = processor.amount
            st.session_state["result_img"] = processor.result_img
            st.session_state["raw_img"] = processor.raw_img
            st.session_state["annotations"] = processor.annotations

    if st.session_state["recognized"]:
        st.metric("💵 合計金額", f"{st.session_state['amount']} 円")

        with st.expander("📌 アノテーションを表示"):
            st.code("\n".join(st.session_state["annotations"]), language="")

        # 保存ボタン
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if st.button("🔽 画像を保存"):
            save_path = os.path.join(log_dir, f"recognized_coin_{timestamp}.png")
            save_path_raw = os.path.join(log_dir, f"raw_img_{timestamp}.png")
            save_path_txt = Path(log_dir) / f"raw_img_{timestamp}.txt"
            cv2.imwrite(save_path, st.session_state["result_img"])
            cv2.imwrite(save_path_raw, st.session_state["raw_img"])
            with open(save_path_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(st.session_state["annotations"]))
            st.info(f"画像とアノテーションデータを保存しました: {save_path}")

        # 再認識ボタン
        if st.button("🔄 再認識"):
            st.session_state["recognized"] = False
            st.session_state["amount"] = None
            st.session_state["result_img"] = None
            new_key = f"example_{int(time.time())}"
            ctx = webrtc_streamer(
                key=new_key,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False}
            )
            st.stop()
