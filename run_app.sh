#!/bin/bash

# Create necessary directories
mkdir -p data

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit non trovato. Installazione requisiti in corso..."
    echo "Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Avvio applicazione Ottimizzazione Spazi Espositivi..."
echo "Starting Exhibition Space Optimization app..."
streamlit run app/app.py 