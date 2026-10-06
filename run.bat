@echo off
REM Starts CareerGuide AI (run from the project folder after installing dependencies)
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Run: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
