# Dashboard for interpreting credit granting predictions: structure, model, customized metrics and interpretability

This project aims at deploying a dashboard to consult predictions for the granting of the credit (default or not) of about 300k clients.
The dataset comes from the Kaggle competition.
## Table of content

1. Structure of the projet
2. Selection and training of the model
3. customized cost function and optimization
4. Interpretability of the model using SHAP

## 1. Structure of the projet

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
### a. GridSearchCV
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
  - RFE(estimator=DecisionTreeClassifier(), n_features_to_select=50)
  - SelectKBest(k=50)

The grid of parameters for each step is available in the notebook P7_modelling
### b. Best pipe
Given that the target class is unbalanced (default 9% VS non-default 91%), AUC is preferred over accuracy. The following 
pipeline achieved the best score of 0.74:
- imputer: SimpleImputer(strategy='median')
- resampling: RandomUnderSampler(sampling_strategy='majority')  
- feature_selection: SelectKBest(k=50)
- model: LGBMClassifier(colsample_bytree=0.8, max_depth=7, min_split_gain=0,
                n_estimators=40, num_leaves=10, objective='binary',
                random_state=seed, reg_alpha=0.1, reg_lambda=0, subsample=1)

<p align="center">
  <img width="500" src="https://github.com/vpvinc/P7_loan_scoring_model/blob/assets/model_comparison.PNG?raw=true" />
</p>

## 3. customized cost function and optimization
Using AUC, we determined the best model to maximise both recall and specificity regardless of the probability threshold. 
However, we now want to determine the best threshold in order to minimize the rate of false negatives which represent
the worst error for the bank. To achieve this optimization, we will choose the threshold that maximize the following
score:    
`score = (tp - 100*fn- fp + tn)/(tp + 100*fn + fp + tn)`  
False negatives are penalized a 100 times more than false positive. This factor is an arbitrary choice and should be 
given by the bank.
By plotting the default probability threshold VS this score. We can visually identify the threshold that maximize the 
score.
<p align="center">
  <img width="811" src="https://github.com/vpvinc/P7_loan_scoring_model/blob/assets/scoreVSthr.png?raw=true" />
</p>

A threshold of 30% maximizes the score, which corresponds to a fn rate of 10% and a fp rate of 65%

## 4. Interpretability of the model using SHAP

See this [repo](https://github.com/slundberg/shap/tree/master) by slundberg for more information on SHAP

As stated by slundberg:
> SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. 
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and 
their related extensions.

In simple terms, it's a set of tools that helps interpret and measure the individual effects of features to the output 
of a black-box model. 
One of the most commonly used plot to interpret a single prediction is the waterfall. Here below is a waterfall plot for
a prediction of credit granting:

<p align="center">
  <img width="511" src="https://github.com/vpvinc/P7_loan_scoring_model/blob/assets/waterfall_ex.PNG?raw=true" />
</p>

The above explanation shows features each contributing to push the model output from the base value (the average model 
output over the training dataset we passed) to the model output. The base value above is 0.002 (the average default 
probability) Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.  

Another way to visualize the same explanation is to use a force plot. Here below is an example for the same prediction 
as above:

<p align="center">
  <img width="511" src="https://github.com/vpvinc/P7_loan_scoring_model/blob/assets/forceplot_ex.PNG?raw=true" />
</p>

To get an overview of which features are most important for a model we can plot the SHAP values of every feature for 
every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values 
to show the distribution of the impacts each feature has on the model output. The color represents the feature value 
(red high, blue low). This reveals for example that a low EXT_SOURCE_2_EXT_SOURCE_3 value increases the default 
probability.

<p align="center">
  <img width="511" src="https://github.com/vpvinc/P7_loan_scoring_model/blob/assets/summaryplot_ex.PNG?raw=true" />
</p>