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
ROOT_DIR = Path(__file__).parent
DATA_YAML_PATH = ROOT_DIR / "datasets" / "data.yaml"
PROJECT_BASE = ROOT_DIR / "finetuning_result"
MODEL_PATH = PROJECT_BASE /"yolo11n"/"best.pt"
recognizer = CoinRecognizer(model_path=str(MODEL_PATH))

LOG_DIR = ROOT_DIR / "classifier_result"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# セッション初期化
st.session_state.setdefault("recognized", False)
st.session_state.setdefault("amount", None)
st.session_state.setdefault("result_img", None)
st.session_state.setdefault("raw_img", None)
st.session_state.setdefault("annotations", None)
st.session_state.setdefault("stream_started", False)  # ストリーム起動の初期化フラグ

# 動画処理クラス
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_time = 0
        self.interval = 1.0
        self.result_img = None
        self.recognized = False
        self.amount = None
        self.raw_img = None
        self.annotations = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = cv2.getTickCount() / cv2.getTickFrequency()
        output_img = img

        # 認識済みなら結果画像を常に返す（プレビューとして）
        if self.recognized and self.result_img is not None:
            return av.VideoFrame.from_ndarray(self.result_img, format="bgr24")

        if now - self.last_time >= self.interval:
            self.last_time = now
            detected_img, total_amount, annotations, raw_img = recognizer.recognize(img)
            print(f"[recv] detected_img shape={detected_img.shape}, total_amount={total_amount}")

            # 認識成功時
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


# --- タイトル --- 
st.title("💰 硬貨識別ツール")

# --- 左右カラム配置 --- 
col1, col2 = st.columns([2, 1])

with col1:
    # WebRTC ストリーム開始
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1920}, "height": {"ideal": 1080}},
            "audio": False
        }
    )

with col2:
    st.subheader("🔍認識結果")

    # video_processor が存在したら参照
    processor = None
    if ctx.video_processor:
        processor = ctx.video_processor
        # 新しいストリーム開始時に processor の状態をリセットしておく
        if not st.session_state["stream_started"]:
            try:
                processor.amount = None
                processor.result_img = None
                processor.recognized = False
                st.session_state["stream_started"] = True
            except Exception:
                # 安全に無視（たまに race が起きる）
                pass

    # --- 自動再描画（ポーリング） ---
    # streamlit-autorefresh がインストールされていればそれを使って非ブロッキングに再実行
    try:
        from streamlit_autorefresh import st_autorefresh
        # 500ms ごとに自動リフレッシュ（調整可）
        st_autorefresh(interval=500, key="autorefresh")
    except Exception:
        # フォールバック：手動更新ボタン（自動更新が無ければユーザーが更新する）
        st.write("自動更新が利用できません。`pip install streamlit-autorefresh` を推奨します。")
        if st.button("更新"):
            st.experimental_rerun()

    # --- 認識フラグが立っていたらストリームを停止して結果表示へ ---
    if processor is not None and getattr(processor, "recognized", False):
        # WebRTC ストリームを停止（安全に何度呼んでもよい）
        try:
            ctx.stop()
        except Exception:
            # ctx.stop() が例外を出すことは稀なので無視して先へ進む
            pass

        # 結果をセッションステートに移す
        st.session_state["recognized"] = True
        st.session_state["amount"] = getattr(processor, "amount", None)
        st.session_state["result_img"] = getattr(processor, "result_img", None)
        st.session_state["raw_img"] = getattr(processor, "raw_img", None)
        st.session_state["annotations"] = getattr(processor, "annotations", [])

    # --- 右側の UI 表示 ---
    if st.session_state["recognized"]:
        st.metric("💵 合計金額", f"{st.session_state['amount']} 円")

        with st.expander("📌 アノテーションを表示"):
            if st.session_state["annotations"]:
                # 行番号を付ける
                annotated_lines = [
                    f"{i+1:>2}: {line}"  # 右揃えで3桁の行番号を付ける
                    for i, line in enumerate(st.session_state["annotations"])
                ]
                st.code("\n".join(annotated_lines), language="")
            else:
                st.write("アノテーションはありません。")

        # --- 保存ボタン --- 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if st.button("🔽 データ保存"):
            save_path = os.path.join(LOG_DIR, f"recognized_coin_{timestamp}.png")
            save_path_raw = os.path.join(LOG_DIR, f"raw_img_{timestamp}.png")
            save_path_txt = Path(LOG_DIR) / f"raw_img_{timestamp}.txt"
            if st.session_state["result_img"] is not None:
                cv2.imwrite(save_path, st.session_state["result_img"])
            if st.session_state["raw_img"] is not None:
                cv2.imwrite(save_path_raw, st.session_state["raw_img"])
            with open(save_path_txt, "w", encoding="utf-8") as f:
                if st.session_state["annotations"]:
                    f.write("\n".join(st.session_state["annotations"]))
            st.info(f"画像とアノテーションデータを保存しました: {save_path}")

        # --- 再認識ボタン ---
        if st.button("🔄 再認識"):
            # セッションの結果をリセットして新しいストリームを作る
            st.session_state["recognized"] = False
            st.session_state["amount"] = None
            st.session_state["result_img"] = None
            st.session_state["raw_img"] = None
            st.session_state["annotations"] = None
            st.session_state["stream_started"] = False
            # 新しいキーでストリームを再生成（ページ再実行時に col1 側の webrtc_streamer が新しいキーを使って起動する）
