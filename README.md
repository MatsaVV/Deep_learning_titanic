This project aimed to train me on the use of Azure Machine Learning. The resources no longer exist, so it is not usable.

# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using machine learning. The project includes data preprocessing, model training, model evaluation, and deployment on Azure Machine Learning using MLflow for experiment tracking and model management.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [License](#license)

## Project Overview

The main steps in this project include:
1. Data preprocessing and exploratory data analysis (EDA).
2. Training and evaluating machine learning models.
3. Using MLflow to track experiments and manage models.
4. Deploying the best model on Azure Machine Learning.
5. Setting up continuous integration and continuous deployment (CI/CD) with GitHub Actions.

## Directory Structure

├── .github
│ ├── workflows
│ │ └── ci-cd-pipeline.yml
├── data
│ └── Titanic.csv
├── notebooks
│ └── eda.ipynb
├── src
│ ├── train.py
│ ├── score.py
│ ├── config.json
│ └── requirements.txt
├── models
│ └── best_model.pkl
├── README.md
└── .gitignore

## License

This project is licensed under the MIT License.
