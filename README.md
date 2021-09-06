# Dashboard for interpreting credit granting predictions with SHAP

---
---
This project aims at deploying a dashboard to consult predictions for the granting of the credit (default or not) of about 300k clients.
The dataset comes from the Kaggle competition.
## Table of content

---
1. Structure of the projet
2. Selection and training of the model
3. customized cost function and optimization
4. Interpretability of the model using SHAP

## 1. Structure of the projet

---
**This project articulates around 12 files:**
- P7_EDA: notebook for EDA (adapted from [Will Koehrsen](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction) 's work)
- P7_modelling: notebook containing:
  - preprocessing steps 
  - GridSearchCV for two models, 
  - optimization of the best model with a custom metrics, 
  - computing of a Tree explainer and SHAP values
  - exportation of data, explainer and model to be used with Streamlit
- main.py: application file to be run with streamlit. The user must type in an ID. The prediction (default/not default) 
and the default probability are then displayed along with SHAP plots (waterfall, force-plot, summary plot)
- helper.py: package containing functions used in main.py
- prep_train.csv: dataset preprocessed. This file is not on the repo as per Kaggle policy
- train.csv: dataset unprocessed (only imputation by median and mode). This file is not on the repo as per Kaggle policy
- folder "data":
  - explainer_shapvs.pkl: tree_explainer fitted with best model and shap values computed for the whole dataset. Both are 
pickled together and use in main.py. The file is not available on Github due to size constraints
  - pipe.pkl: best pipe pickled used in main.py
  - num_cat_cols: pkl file containing lists used to display the last graph in the dashboard
- environment.yml: file to set up dependencies with conda
- requirements.txt: file to set up dependencies with pip

## 2. Selection and training of the model

---
Two high performing classifiers were considered here: Random Forest Classifier (RFC) and Light GBM (LGBM).
A cross-validated GridSearch was applied for these two model with the following grid for preprocessing steps:  
- imputer: 
  - SimpleImputer(strategy='median')
  - SimpleImputer(strategy='mean')
  - SimpleImputer(strategy='constant')
- resampling: 
  - SMOTE(sampling_strategy=0.8)
  - RandomUnderSampler(sampling_strategy='majority')  
- feature_selection: 
  - RFE(estimator=DecisionTreeClassifier()
  - n_features_to_select=50), SelectKBest(k=50)

The grid of parameters for each step is available in the notebook P7_modelling

## 3. customized cost function and optimization

---


## 4. Interpretability of the model using SHAP

---

