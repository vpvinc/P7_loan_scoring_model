requirement.txt est automatiquement récupéré ?

# importing train data
train = pd.read_csv("C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/train.csv")

# unpickling explainer, shap values an model
with open("C:/Users/VP/Google Drive/Education/OC/working_directory/P7/Credit_granting_CS_Streamlit/explainer_shapvs", 'rb') as fic:
    mon_depickler = pickle.Unpickler(fic)
    loaded = mon_depickler.load()

explainer, shapvs = loaded

# defining the function for the custom threshol function


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