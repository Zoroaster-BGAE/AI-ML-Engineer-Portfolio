@echo off
REM One-click launcher: creates a local virtual environment (first run only),
REM installs the requirements into it, then starts the Streamlit app.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating virtual environment in .venv ...
    python -m venv .venv || goto :fail
    echo [2/2] Installing requirements ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo Starting the app - it opens in your browser at http://localhost:8501
echo Press Ctrl+C in this window to stop it.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:fail
echo.
echo Setup failed. Make sure Python 3.10+ is installed and on PATH, then delete the .venv folder and try again.
pause
exit /b 1
