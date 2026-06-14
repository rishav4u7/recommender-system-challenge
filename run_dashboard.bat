@echo off
echo ===================================================
echo   AdRec Intelligence Dashboard Launcher (Windows)
echo ===================================================
echo.
echo Checking and installing dependencies...
python -m pip install pandas numpy scipy scikit-learn streamlit
echo.
echo Starting Streamlit Dashboard...
python -m streamlit run app.py
pause
