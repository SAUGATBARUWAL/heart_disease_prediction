# ❤️ Heart Disease Prediction using KNN and Streamlit

## 📌 Project Overview

This project is a Machine Learning web application that predicts the likelihood of heart disease based on a patient's medical information. The model is built using the **K-Nearest Neighbors (KNN)** algorithm and deployed through an interactive **Streamlit** web interface.

The application allows users to enter various health-related parameters and receive an instant prediction regarding heart disease risk.

---

## 🚀 Features

- User-friendly Streamlit web interface
- Heart disease risk prediction in real time
- Machine Learning model based on K-Nearest Neighbors (KNN)
- Data preprocessing and feature scaling
- Fast and lightweight deployment
- Easy-to-understand prediction results

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

## 📊 Dataset

The project uses the **Heart Disease Dataset**, which contains patient health records and medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Slope of Peak Exercise ST Segment

Target Variable:
- **0** → No Heart Disease
- **1** → Heart Disease Present

---

## ⚙️ Machine Learning Workflow

1. Load and explore the dataset.
2. Handle missing values (if any).
3. Perform feature scaling using StandardScaler.
4. Split data into training and testing sets.
5. Train the KNN classification model.
6. Evaluate model performance using accuracy metrics.
7. Save the trained model.
8. Deploy the model using Streamlit.

---

## 🧠 K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification and regression problems. It predicts the class of a new data point by finding the K nearest data points in the training dataset and selecting the most common class among them.

### Advantages

- Simple and easy to understand
- No assumptions about data distribution
- Effective for small and medium-sized datasets

### Limitations

- Computationally expensive for very large datasets
- Sensitive to feature scaling
- Performance depends on the choice of K

---

## 📈 Model Performance

Example Metrics:

- Accuracy: 85% - 95%
- Precision: High
- Recall: High

*Note: Actual results may vary depending on the dataset and hyperparameter tuning.*

---

## ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📂 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── knn_heart_disease_model.pkl
├── scaler.pkl
├── heart.csv
├── model_columns.okl
├── heartdisease.ipynb

```


---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a feature branch, and submit a pull request.

-