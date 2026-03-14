# AI-CRAFT-PROJECT

### PCOS Risk Prediction using Machine Learning with Web Interface & Chatbot

## Project Overview

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting many women. Early detection can help manage symptoms and reduce long-term health risks.

This project uses **machine learning models** to predict the likelihood of PCOS based on health indicators and symptoms. The system includes a **web interface and chatbot** that allow users to interact with the model and understand potential risk factors.

The goal is to demonstrate how **data science, machine learning, and web technologies** can work together to build a practical healthcare assistance tool.

---

## Features

* Data preprocessing and cleaning pipeline
* Multiple machine learning models for prediction
* Model performance comparison
* Web-based interface for prediction input
* Interactive chatbot support
* Visualization of model performance (ROC curves)

---

## Machine Learning Models Used

The following models are trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

These models are compared to determine which performs best for predicting PCOS risk.

---

## Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Web Application

* Stremlit

### Visualization

* Matplotlib
* Plotly


## Project Structure

```
AI-CRAFT-PROJECT
│
├── Data
│   └── Dataset used for training and testing
│
├── models
│   └── Trained machine learning models
│
├── notebooks
│   └── Jupyter notebooks for experimentation
│
├── src
│   ├── chatbot.py
│   ├── preprocessing.py
│   ├── train_LR.py
│   ├── train_RF.py
│   ├── train_XG.py
│   └── model_compare.py
│
├── app.py
│   └── Web application interface
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
```
## How to Run the Project

### 1. Clone the repository

```
git clone <repository-url>
cd AI-CRAFT-PROJECT
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

### 4. Open the web interface



---

## Future Improvements

* Improve chatbot intelligence
* Add explainable AI for model predictions
* Deploy the web application online
* Improve UI/UX of the interface
* Add more healthcare insights and recommendations

---

## Contributors

* **Meihul Saini** – Web interface, chatbot integration
* **Anjali** – Machine learning model development

---

## Disclaimer

This project is for **educational and research purposes only**.
It should **not be used as a substitute for professional medical advice or diagnosis**.


