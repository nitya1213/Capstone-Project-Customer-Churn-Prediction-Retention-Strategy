#!/usr/bin/env bash

# ============================
# Customer Churn Project Runner
# ============================

echo "Checking if virtual environment exists..."
if [ ! -d "venv" ]; then
    echo "⚙️  Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate
echo "🔹 Checking dependencies..."

echo "🔹 Checking dependencies..."

if pip install -r requirements.txt --dry-run &> /dev/null; then
    echo "✅ All dependencies already satisfied."
else
    echo "⚙️ Missing or outdated dependencies – installing..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi


echo "Running EDA..."
python3 src/eda_cases.py

echo "Running hyperparameter tuning..."
python3 src/tuning.py

echo "Training & evaluating final models..."
python3 src/modeling.py

echo "Done! Check outputs/, models/ for results."
