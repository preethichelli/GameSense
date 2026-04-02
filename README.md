# 🎮 GameSense

## Overview
GameSense is a simple machine learning web app that analyzes Steam game data and predicts whether a game is likely to be successful. It allows users to explore the dataset, apply filters, and make predictions using an interactive interface.

## Features
- Search games by name  
- Filter by genre, price, and reviews  
- View basic visualizations  
- Predict game success using ML  

## Machine Learning
- Model: Random Forest Classifier  
- Target: Game success based on review score  
- Features used:
  - Price_USD  
  - Discount_Pct  
  - Review_Score_Pct  
  - Total_Reviews  
  - 24h_Peak_Players  

## Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
