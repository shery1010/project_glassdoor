🔍 Project Title: Job Salary Prediction Using Machine Learning
📌 Table of Contents
Project Overview

Problem Statement

Dataset

Tech Stack Used

Modeling Approach

Results

How to Run

Future Work

Contributors

✅ Project Overview
This project aims to predict the salaries of jobs based on their descriptions and company information using machine learning regression models. The goal is to help candidates and analysts understand the factors that influence job compensation.

❓ Problem Statement
Companies list job descriptions without specifying salary details. This project builds a regression model to predict salaries based on features like job title, location, company size, industry, and job description text.

📂 Dataset
The dataset contains job postings with the following features:

Job Title

Location

Company Info (Size, Revenue, etc.)

Job Description

Salary Estimate (Target)

🛠️ Tech Stack Used
Python

Pandas, NumPy, Scikit-learn

NLTK (for NLP preprocessing)

Matplotlib, Seaborn (for EDA)

Jupyter Notebook / Google Colab

📈 Modeling Approach
Data Preprocessing:

Cleaned job descriptions (removed stopwords, punctuation, lemmatized text).

Handled missing values and outliers.

Categorical encoding using One-Hot Encoding.

Feature Engineering:

Extracted keywords from job descriptions.

POS tagging applied.

Model Building:

Random Forest Regressor with GridSearchCV and RandomizedSearchCV.

Gradient Boosting Regressor with hyperparameter tuning.

Evaluation Metrics: MAE, MSE, RMSE, R²

🔮 Future Work
Improve text features using TF-IDF or BERT embeddings.

Integrate web scraping for live job postings.

Build a web app using Flask/Streamlit for live predictions.

