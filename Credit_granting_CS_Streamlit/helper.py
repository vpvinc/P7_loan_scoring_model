"""module to store helpful functions classes used in main.py"""

# import numpy as np
import pandas as pd
# import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import dill
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
seed=42

# functions to load data and objects

# importing and unpickling train df, explainer, shap values and pipe
# defining functions to do so
@st.cache
def load_raw_application_train_data():
    """returns train dataset"""
    return pd.read_csv(
        # "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/train.csv").iloc[:, 1:]  # skip Unamed column
        "https://github.com/vpvinc/P7_loan_scoring_model/blob/main/train.csv?raw=true").iloc[:, 1:]

@st.cache
def load_prep_train_data():
    """returns train dataset preprocessed (excluding Random Under Sampling). Indeed we need preprocessed data for all
    IDs, not just the ones undersampled to display SHAP plots. Same logic as for the computation of Shap values for
    all IDs"""
    return pd.read_csv(
        # "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/prep_train.csv").iloc[:, 1:]  # skip Unamed
        "https://github.com/vpvinc/P7_loan_scoring_model/blob/main/prep_train.csv?raw=true").iloc[:, 1:]

def load_num_cat_cols():
    """returns tuple (num_cols, cat_cols)"""
    with open(
            # "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/data/num_cat_cols.pkl",
            "https://github.com/vpvinc/P7_loan_scoring_model/blob/main/Credit_granting_CS_Streamlit/data/num_cat_cols.pkl?raw=true",
            "rb") as file:
        return dill.load(file)

def load_explainer_shapvs():
    """returns tuple (fitted explainer, shap values)"""
    with open(
            # "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/data/explainer_shapvs.pkl",
            "https://github.com/vpvinc/P7_loan_scoring_model/blob/main/Credit_granting_CS_Streamlit/data/explainer_shapvs.pkl?raw=true",
            "rb") as file:
        return dill.load(file)


def load_pipe():
    """returns fitted pipe"""
    with open(
            # "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/data/pipe.pkl",
            "https://github.com/vpvinc/P7_loan_scoring_model/blob/main/Credit_granting_CS_Streamlit/data/pipe.pkl?raw=true",
            "rb") as file:
        best_pipe = dill.load(file)
    return best_pipe

# functions for predictions

# defining the function for the custom threshol function
def cust_predict_proba(pos_class_proba_array, thres=0.3):
    """This is a threshold function that convert an array of probabilities of belonging to class "1" into a array of
    labels 0 or 1 with respect to the threshold. We will use this function to set up an optimal threshold that minimize
    false negative, unlike the default value of 0.5"""
    pos_class_proba_array = pd.Series(pos_class_proba_array)
    pos_class_proba_array = pos_class_proba_array.map(lambda x: 1 if x > thres else 0)
    return pos_class_proba_array.to_numpy()


# defining function to display default instead of 1
def display_pred(prediction):
    if prediction == 0:
        return 'No default'
    else:
        return 'Default'

# functions for plotting

def set_style_pers():
    """set default sns style and customize rc params"""

    from cycler import cycler
    matplotlib.rcParams.update(
        {
            'axes.titlesize': 25,  # axe title
            'axes.labelsize': 20,
            'axes.prop_cycle': cycler('color',
                                      ['#0D8295', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                       '#7f7f7f', '#bcbd22', '#17becf']),  # lines colors
            'lines.linewidth': 3,
            'lines.markersize': 150,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'font.family': 'Century Gothic'
        }
    )

# defining function to display shap force plot in streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# display summary plot for all clients:
@st.cache(suppress_st_warning=True, hash_funcs={dict: lambda _: None})
def summary_plot_all(shapvs, prep_train, target):
    shapvs_sample, *_ = train_test_split(shapvs[1], target, test_size=0.9, stratify=target,
                                         random_state=seed)
    prep_train_sample, *_ = train_test_split(prep_train.iloc[:, :-1], target, test_size=0.9, stratify=target,
                                               random_state=seed)
    dico_cache = dict()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shapvs_sample, prep_train_sample.values, prep_train_sample.columns, max_display=20)
    dico_cache['fig'] = fig
    return dico_cache

# function for quantitative features
def kde_plot_num_var(data=None, x=None, y=None):
    """function to display the boxplots of default and non-deafult clients for a quantitative feature
    params:
    -data: DataFrame: train
    -x: str: Default(1)/non-default(0)
    -y: str: feature considered"""
    set_style_pers()
    sns.boxplot(data=data, x=x, y=y, showfliers=False)
    plt.title("Average value of feature between 'default' and 'non default' clients",  fontsize=16)
    plt.ylabel('{}'.format(y))

# function to be used for the barplot of qualitative features
def barPerc(df,xVar,ax):
    '''
    barPerc(): Add percentage for hues to bar plots
    args:
        df: pandas dataframe
        xVar: (string) X variable
        ax: Axes object (for Seaborn Countplot/Bar plot or
                         pandas bar plot, hue must be specified
                         a parameter of the plot)
    '''
    set_style_pers()
    # 1. how many X categories
    ##   check for NaN and remove
    numX=len([x for x in df[xVar].unique() if x==x])

    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    ## 2a. For each X variable
    for ind in range(numX):
        ## 2b. Get every hue bar
        ##     ex. 8 X categories, 4 hues =>
        ##    [0, 8, 16, 24] are hue bars for 1st X category
        hueBars=bars[ind:][::numX]
        ## 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hueBars])

        # 3. Print the percentage on the bars
        for bar in hueBars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height(),
                    f'{bar.get_height()/total:.0%}',
                    ha="center",va="bottom")

#barplot for qualitative features
def bar_plot_cat_var(data=None, x=None, hue=None):
    barPerc(data,
            x,
            sns.countplot(data=data,x=x, hue=hue))
    plt.xticks(rotation=90)
    plt.title("Proportion of clients 'default' and 'non default' per modality of the feature",  fontsize=16)
    plt.ylabel('Number of clients')