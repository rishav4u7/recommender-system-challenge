#!/bin/bash
echo "==================================================="
echo "  AdRec Intelligence Dashboard Launcher (Unix)"
echo "==================================================="
echo ""
echo "Checking and installing dependencies..."
python3 -m pip install pandas numpy scipy scikit-learn streamlit
echo ""
echo "Starting Streamlit Dashboard..."
python3 -m streamlit run app.py
