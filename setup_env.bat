@echo off

echo ���z�����쐬
SET ENV_NAME=.env_coin_classifier
py -3.10 -m venv %ENV_NAME%

echo ���z�����A�N�e�B�x�[�g
call %ENV_NAME%\Scripts\activate.bat


echo ���C�u�����̃C���X�g�[��
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ���z�� %ENV_NAME% ���쐬���܂����B
pause