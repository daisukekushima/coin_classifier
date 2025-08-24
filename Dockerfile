# Python 3.11 をベースイメージとして使用
FROM python:3.11

# 環境変数を設定
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストールするためのファイルをコピー
COPY requirements.txt .

# OpenCVのネイティブ依存ライブラリをインストール（Debian Trixie 対応）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# pipをアップグレードし、Pythonライブラリをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# Streamlitが使用するポートを公開
EXPOSE 8501

# アプリケーションの起動コマンド
CMD ["streamlit", "run", "streamlit_run.py"]
