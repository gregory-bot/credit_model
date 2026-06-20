#!/bin/bash
set -e

echo "Starting Credit Scoring API..."

# Check if model exists
if [ ! -f models/best_model.pkl ]; then
    echo "ERROR: Model not found! Train the model first."
    echo "Run: python run_pipeline.py --target fpd_fmd"
    exit 1
fi

# Start the API
uvicorn credit_scoring.api:app --host 0.0.0.0 --port 
