@echo off

echo 仮想環境を作成
SET ENV_NAME=.env_coin_classifier
if not exist %VENV_NAME%\Scripts\activate (
    py -3.11 -m venv %ENV_NAME%
    echo 仮想環境を作成しました。
) else (
    echo 仮想環境は既に存在します。
)


echo 仮想環境をアクティベート
call %ENV_NAME%\Scripts\activate.bat


echo ライブラリのインストール
pip install -r requirements.txt

echo 仮想環境 %ENV_NAME% を作成しました。
pause

