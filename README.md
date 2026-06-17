# 🏠 House Price Prediction

A machine learning web application that predicts California housing prices based on user inputs.

🔗 **Live Demo:** https://house-price-prediction-fhqy.onrender.com/

---

## 📌 Features
- Predicts house price based on location, building age, and room count
- Shows estimated price range based on model error margin
- Responsive design (mobile & desktop)

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **ML Model:** Scikit-learn (Random Forest Regressor)
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS
- **Deployment:** Render.com

---

## 📊 Model Performance
- R² Score: 0.80
- RMSE: 0.50
- Dataset: California Housing (sklearn) — 20,640 samples

---

## 🚀 How to Run Locally
git clone https://github.com/Parsa-ux360/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python train_model.py
python app.py


---

## 📁 Project Structure
house-price-prediction/
├── app.py              # Flask backend
├── train_model.py      # ML model training
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    └── index.html


---

Built by **Parsa Shahi**