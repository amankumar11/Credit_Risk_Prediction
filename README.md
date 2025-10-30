# 💳 Credit Risk Analyzer  
**Predict Loan Default Risk using Machine Learning**

This project is a **Streamlit web application** that predicts whether a loan applicant is likely to **default** on their loan or not, based on financial and demographic information.  
It uses a trained **Random Forest Classifier** model on a Kaggle dataset: *Loan Default Prediction Dataset*.  

---

## Features

- Predicts **loan default risk** from user inputs  
- Handles **categorical encoding and scaling** automatically  
- Displays prediction confidence and risk category  
- Includes a **trained model** (`credit_analyzer_model.pkl`)  
- Built with **Streamlit**, **scikit-learn**, and **pandas**

---

## Setup Instructions

### Clone this repository
```bash
git clone https://github.com/yourusername/credit-risk-analyzer.git
cd credit-risk-analyzer
```

### Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the streamlit app
```bash
streamlit run app.py
```

----

## Model Details

The model was trained using a RandomForestClassifier with:

- n_estimators=200

- max_depth=10

- class_weight='balanced'

Preprocessing includes:

- Label encoding for categorical columns

- Standard scaling for numerical features

- Handling missing values and duplicates
