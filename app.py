import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset, model, and vectorizer
@st.cache_data
def load_data_and_model():
    # Load dataset
    dataset_path = 'drugsComTrain_raw.tsv'  # Replace with the actual dataset path
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Load trained model
    model_path = 'model.pkl'  # Replace with the actual model file path
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load TF-IDF vectorizer
    vectorizer_path = 'vectorizer.pkl'  # Replace with the actual vectorizer file path
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return df, model, vectorizer

# Load the data and model
df, model, vectorizer = load_data_and_model()

# Function to extract top drugs
def top_drugs_extractor(condition, model, vectorizer, df):
    # Transform user input
    transformed_condition = vectorizer.transform([condition])
    
    # Predict condition using the model
    predicted_condition = model.predict(transformed_condition)[0]
    
    # Extract top drugs for the predicted condition
    top_drugs = (
        df[df["condition"].str.contains(predicted_condition, case=False, na=False)]
        .groupby("drugName")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    return top_drugs

# Streamlit UI
st.title("Drug Recommendation System")

st.header("Enter Health Condition")
user_input = st.text_input("Describe your health problem (e.g., 'Depression', 'Anxiety'):")

if st.button("Get Top Drugs"):
    if user_input:
        try:
            top_drugs = top_drugs_extractor(user_input, model, vectorizer, df)
            st.success(f"Top Recommended Drugs for '{user_input}':")
            for i, drug in enumerate(top_drugs, 1):
                st.write(f"{i}. {drug}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a health condition.")