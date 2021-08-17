
import streamlit as st
import requests


def main():

    st.title("Spam Classification")
    message = st.text_input('Enter Text to Classify')

    if st.button('Predict'):
        payload = {
            "text": message
        }
        
        # this is the below line that I need to adapt to return the result for the credit + the waterfall of SHAP values
        res = requests.post(f"http://service:8000/predict/",json=payload )
  
        with st.spinner('Classifying, please wait....'):
            st.write(res.json())




if __name__ == '__main__':
    main()