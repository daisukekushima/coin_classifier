@echo off

echo 仮想環境を作成
SET ENV_NAME=.env_coin_classifier
py -3.10 -m venv %ENV_NAME%

echo 仮想環境をアクティベート
call %ENV_NAME%\Scripts\activate.bat


echo ライブラリのインストール
python -m pip install --upgrade pip
pip install -r requirements.txt

echo 仮想環境 %ENV_NAME% を作成しました。
pause