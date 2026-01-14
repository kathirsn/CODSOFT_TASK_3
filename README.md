🌸 Iris Flower Classification using Machine Learning

📌 Project Overview
This project implements a machine learning classification model to predict the species of an Iris flower based on its physical measurements. It uses the classic Iris dataset and a Random Forest Classifier to achieve high prediction accuracy.

This project was completed as part of CodSoft – Machine Learning Task 3 and demonstrates a complete ML workflow including data loading, model training, cross-validation, model saving, and making predictions on new data.

Author: Kathir S N
Batch: December Batch B70
Domain: Data Science

📊 Dataset Information
The Iris dataset contains 150 samples with four numerical features:

Sepal Length (cm) – Length of the sepal
Sepal Width (cm) – Width of the sepal
Petal Length (cm) – Length of the petal
Petal Width (cm) – Width of the petal

🎯 Target Classes
0 → Iris-setosa
1 → Iris-versicolor
2 → Iris-virginica

🛠️ Technologies Used
Python 3.12
Pandas – Data handling
NumPy – Numerical computation
Scikit-learn – Machine learning
Joblib – Model persistence
VS Code – Development environment

🧠 Machine Learning Model
RandomForestClassifier

Why Random Forest?
It handles non-linear relationships well, reduces overfitting, and provides high accuracy with minimal tuning.

📈 Model Evaluation
The model was evaluated using cross-validation.

Cross-validation scores: [0.9666, 0.9666, 0.9333, 0.9666, 1.0]
Mean Accuracy: 96.66%

The model performs consistently and reliably across all folds.

💾 Saving the Model
The trained model is saved using Joblib so it can be reused without retraining.
Saved file name: iris_classifier.pkl

🔮 Making Predictions
A sample flower with measurements [5.1, 3.5, 1.4, 0.2] was passed to the model for prediction.
The predicted class was:

Predicted class: 0

This corresponds to Iris-setosa.

📂 Project Structure

TASK3_CODSOFT.ipynb
iris_classifier.pkl
README.md

🚀 How to Run the Project

Download or clone the project

Open the notebook in VS Code or Jupyter Notebook

Install required libraries: pandas, numpy, scikit-learn, joblib

Run all cells to train, evaluate, save, and test the model

🧪 Key Learnings
Built a supervised classification model
Used cross-validation for reliable evaluation
Saved and reused trained ML models
Handled feature-name warnings correctly
Maintained a clean and professional project structure

⭐ Acknowledgements
Scikit-learn Documentation
UCI Machine Learning Repository
CodSoft Internship Program

📝 Conclusion
This project demonstrates an end-to-end machine learning pipeline for a classification task. The Random Forest model achieves high accuracy and is suitable for real-world applications and further deployment.
