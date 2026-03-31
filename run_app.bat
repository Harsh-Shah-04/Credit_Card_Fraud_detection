@echo off
REM Credit Card Fraud Detection - Streamlit App Launcher
REM This script automatically sets up and runs the Streamlit application

echo.
echo ===============================================
echo Credit Card Fraud Detection System
echo Streamlit Application Launcher
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

echo [1/4] Python version:
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo.
    echo [2/4] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [4/4] Installing/updating dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.
echo Starting Streamlit application...
echo The app will open in your default browser.
echo.
echo Press Ctrl+C to stop the server at any time.
echo.

REM Check if models exist
if not exist "models\ann_model.h5" (
    echo WARNING: Model files not found in models/ directory!
    echo Please ensure the following files exist:
    echo   - models/ann_model.h5
    echo   - models/best_ml_model.pkl
    echo   - models/scaler.pkl
    echo   - models/ann_threshold.pkl
    echo.
)

REM Run Streamlit app
streamlit run streamlit_app.py

pause
