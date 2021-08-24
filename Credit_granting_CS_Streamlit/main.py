# run pip install -r requirements.txt in CP

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import shap
import dill
import streamlit as st
import streamlit.components.v1 as components


# importing and unpickling train df, explainer, shap values and model
@st.cache
def load_train_data():
    """returns train dataset"""
    return pd.read_csv(
        "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/train.csv").iloc[:, 1:]  # skip Unamed column


@st.cache
def load_prep_train_data():
    """returns train dataset preprocessed. Indeed the pipeline select 50 features, and we need the names of these 50
    columns to display feature' names in waterfall shap graph"""
    return pd.read_csv(
        "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/prep_train.csv").iloc[:, 1:]  # skip Unamed


def load_explainer_shapvs():
    """returns tuple (fitted explainer, shap values)"""
    with open(
            "C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/explainer_shapvs"
            ".pkl",
            "rb") as file:
        return dill.load(file)


def load_model():
    """returns fitted model"""
    with open("C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/model.pkl",
              "rb") as file:
        best_model = dill.load(file)
    return best_model


# defining the function for the custom threshol function
def cust_predict_proba(pos_class_proba_array, thres=0.5):
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


# defining function to display shap force plot in streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# title
st.title("Credit default probability per client")
st.subheader("The prediction is then analysed by breaking down most influential variables for the prediction")

# Collecting input fed by user
with st.form(key='my_form'):
    client_id = st.text_input(label='Enter client SK_CURR_ID to assess')
    # client_id = int(client_id)
    submit_button = st.form_submit_button(label='Submit')

# condition prediction to submit button
if submit_button:
    # load data
    train = load_train_data()
    prep_train = load_prep_train_data()
    explainer, shapvs = load_explainer_shapvs()
    model = load_model()
    # get index of client SK_CURR_ID
    index_client_id = train['SK_ID_CURR'].index[train['SK_ID_CURR'] == int(client_id)][0]
    # make the default probability
    proba_def = model.predict_proba(train.iloc[index_client_id, 1:-1].values.reshape(1, -1))[0][1] # skip SK_CURR_ID column
    # make the default prediction
    cust_prediction = cust_predict_proba(proba_def, thres=0.5)
    st.write("Prediction on credit refund: {}".format(display_pred(cust_prediction)))
    st.write("Probability of default: {:.1%}".format(proba_def))
    # waterfall graphs display
    # st_shap(shap.force_plot(explainer.expected_value, shapvs[0, :], train.iloc[index_client_id, 1:-1]))
    # st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
    #                                               shapvs[1][index_client_id],
    #                                               features=prep_train.iloc[index_client_id].values,
    #                                              feature_names=prep_train.columns))

    # visualize the prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(shap.force_plot(explainer.expected_value[1], shapvs[1][index_client_id], prep_train.iloc[index_client_id]))

    # Summary plot 1 SHAP
    st.subheader('Summary Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shapvs[1],  prep_train.values, prep_train.columns, max_display=30)
    st.pyplot(fig)
