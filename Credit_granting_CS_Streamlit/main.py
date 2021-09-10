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
from helper import*
import st_state_patch
seed=seed

# set page config
st.set_page_config(page_title="Loan_scoring_dashboard", # or None
                          page_icon="U+1F3E6", # or None
                          layout='wide', # or 'centered' for wide margins
                          initial_sidebar_state='auto')

# variable to display average graphs only once shap graphs are displayed
display_avg = False

# title
st.title("Credit default probability per client")
st.subheader("The prediction is then analysed by breaking down most influential variables for the prediction")

# load data
with st.spinner('Loading data and prediction model...'):
    train = load_raw_application_train_data()
    prep_train = load_prep_train_data()
    num_cols, cat_cols = load_num_cat_cols()
    explainer, shapvs = load_explainer_shapvs()
    pipe = load_pipe()
    model = pipe.named_steps['LBMC']

# Collecting client id to assess in sidebar

with st.sidebar.form(key='my_form'):
    client_id = st.text_input(label='Enter client SK_ID_CURR to assess')
    submit_button = st.form_submit_button(label='Submit')

# command to keep state of the buttons saved. Necessary to prevent streamlit from refreshing between two buttons hit
s = st.State()
if not s:
    s.pressed_first_button = False

# condition prediction to submit button
if submit_button or s.pressed_first_button:
    # asserting if the input ID is correct
    if int(client_id) not in prep_train['SK_ID_CURR'].values:
        # setting up big fontsize
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        # display error message
        st.markdown('<p class="big-font">/!\ ERROR /!\ This ID is either mispelled or not in the database. Example of ID'
                    ': 100004</p>', unsafe_allow_html=True)
    else:
        # preserve the info that you hit a button between runs
        s.pressed_first_button = True
        # get index of client SK_CURR_ID
        index_client_id = prep_train['SK_ID_CURR'].index[prep_train['SK_ID_CURR'] == int(client_id)][0]
        # make the default probability
        proba_def = model.predict_proba(prep_train.iloc[index_client_id, :-1] # skip SK_CURR_ID last column
                                        .values.reshape(1, -1))[0][1]
        # make the default prediction
        cust_prediction = cust_predict_proba(proba_def, thres=0.5)
        # setting up font size
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        # displaying message
        st.markdown('<p class="big-font">Prediction on credit refund: <b>{}<b/> </p>'
                    .format(display_pred(cust_prediction)),
                    unsafe_allow_html=True)
        st.markdown('<p class="big-font">Probability of default: <b>{:.1%}<b/> </p>'.format(proba_def),
                    unsafe_allow_html=True)
        # waterfall graphs display
        st.subheader('Waterfall and force plot')
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                               shapvs[1][index_client_id],
                                               features=prep_train.iloc[index_client_id, :-1].values,
                                               feature_names=prep_train.columns[:-1])
        st.pyplot(fig)
        # visualize the prediction's explanation with force plot (use matplotlib=True to avoid Javascript)
        st_shap(shap.force_plot(explainer.expected_value[1],
                                shapvs[1][index_client_id],
                                prep_train.iloc[index_client_id, :-1]) # excluding last column TARGET
                )
        st.write("The values of features in blue lower the probability of default whereas those in red increase it. \n"
                 "For example, DAYS_EMPLOYED is negative and measure the number of days the client was employed at the date"
                 " of the request. Therefore, a high value for DAYS_EMPlOYED indicates that the client has not been employed"
                 " for long and will be red because it increases the probability of default")
        # Summary plot 1 SHAP
        st.subheader('Summary Plot of all clients')
        with st.spinner('plotting summary of influential variables for all clients...'):
            dico_of_fig = summary_plot_all(shapvs, prep_train, train['Default(1)/No default(0)'])
            st.pyplot(dico_of_fig['fig'])
            st.write("/!\ DAYS_EMPLOYED is negative, so a low value implies a long period of employement")

        display_avg = True

    if display_avg==True: #if shap graph have been displayed
        st.subheader("Value of the client's feature VS the averages of default and non-default clients")
        feature = st.selectbox('Select a feature', sorted(list(train.iloc[:,:-1].columns)))
        st.write("the value of the feature for the client is: {}".format(train.loc[index_client_id, feature]))
        if feature in num_cols:
            fig, ax = plt.subplots(figsize=(25,15))
            kde_plot_num_var(data=train, x='Default(1)/No default(0)', y=feature)
            st.pyplot(fig)

        if feature in cat_cols:
            fig, ax = plt.subplots(figsize=(25,15))
            bar_plot_cat_var(data=train, x=feature, hue='Default(1)/No default(0)')
            st.pyplot(fig)