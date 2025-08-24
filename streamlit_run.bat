@echo off
call .env_coin_classifier\Scripts\activate.bat
streamlit run streamlit_run.py
pause