@echo off

echo ���z�����쐬
SET ENV_NAME=.env_coin_classifier
if not exist %VENV_NAME%\Scripts\activate (
    py -3.11 -m venv %ENV_NAME%
    echo ���z�����쐬���܂����B
) else (
    echo ���z���͊��ɑ��݂��܂��B
)


echo ���z�����A�N�e�B�x�[�g
call %ENV_NAME%\Scripts\activate.bat


echo ���C�u�����̃C���X�g�[��
pip install -r requirements.txt

echo ���z�� %ENV_NAME% ���쐬���܂����B
pause

