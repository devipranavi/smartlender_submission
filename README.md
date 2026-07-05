# Smart Lender — ML-Powered Loan Approval Prediction

## 🎥 Demo Video
[Click here to watch the demo](PASTE_YOUR_VIDEO_LINK_HERE)

## 📌 Project Description
Smart Lender is a machine learning web application that predicts 
whether a bank should approve or reject a loan application.
It trains and compares four ML models (Decision Tree, Random Forest, 
KNN, XGBoost) and deploys the best one (XGBoost ~82% test accuracy) 
through a Flask web application.

## 🛠️ Tech Stack
- Python, Flask, XGBoost, scikit-learn
- pandas, NumPy, Matplotlib, Seaborn
- imbalanced-learn (SMOTE), Jinja2

## 📁 Project Structure
smartlender_submission/
├── app.py                        # Flask backend
├── Loan_Prediction_Model.ipynb   # Full ML pipeline
├── loan_prediction.csv           # Dataset (614 rows)
├── rdf_model.json                # Trained XGBoost model
├── scale1.pkl                    # Fitted StandardScaler
├── requirements.txt              # Dependencies
├── templates/                    # HTML pages
└── static/css/                   # Styling

## ▶️ How to Run
1. Open Anaconda Prompt
2. Navigate to project folder
3. Run: pip install -r requirements.txt
4. Run: python app.py
5. Open http://127.0.0.1:5000 in browser

## 👥 Team Members
| Name | Role |
|---|---|
| Janjanam Devi Pranavi | Team Lead |
| Anusha Vallapu | Member |
| Sanjana Polimetla | Member |
| Kancharla Lakshmi Durga | Member |
| Amarapu Sravana Lakshmi | Member |
