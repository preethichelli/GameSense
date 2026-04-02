🎮 GameSense
📌 Overview

GameSense is an interactive data analytics and machine learning application designed to explore and evaluate Steam game performance. The project focuses on identifying key factors that influence a game's success and provides a predictive model to estimate whether a game is likely to perform well in the market.

🚀 Features

GameSense offers a user-friendly interface where users can:

🔍 Search for specific games and view their details
🎛️ Apply filters based on genre, price range, and review metrics
📊 Analyze trends through visualizations such as price distribution and review patterns
🤖 Predict the success of a game using a trained machine learning model
🧠 Machine Learning

The application uses a Random Forest Classifier to predict game success.
The model is trained on key features such as pricing, user engagement, and review scores.

Input Features:

Price_USD
Discount_Pct
Review_Score_Pct
Total_Reviews
24h_Peak_Players

Output:

Predicts whether a game is likely to be successful or not
▶️ Running the Application

To run the project locally:

pip install -r requirements.txt
streamlit run gamesense.py

The application will launch in your browser, allowing you to interact with the dataset and generate predictions in real time.

🛠️ Technologies Used
Python
Pandas, NumPy
Scikit-learn
Streamlit
