from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import streamlit as st
import pandas as pd
import numpy as np

# Read data and remove duplicates for 'Item', 'Color', 'Style', 'Gender'
data = pd.read_csv("/home/kali/Desktop/fashion/fashionreccomendation.csv")

# Impute missing values in categorical columns (replace with appropriate strategies)
imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical columns
data_imputed = imputer.fit_transform(data[['Item_Name', 'Gender', 'Color', 'Style']])
data_imputed = pd.DataFrame(data_imputed, columns=['Item_Name', 'Gender', 'Color', 'Style'])

# Dynamically get unique categories from the data
gender_categories = data_imputed['Gender'].unique()
color_categories = data_imputed['Color'].unique()
style_categories = data_imputed['Style'].unique()
item_categories = data_imputed['Item_Name'].unique()

# Specify all possible categories for the OneHotEncoder dynamically
encoder = OneHotEncoder(drop='first')
encoded_data = encoder.fit_transform(data_imputed)
print(encoded_data.shape)

# Prepare data for cosine similarity
X = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Item_Name', 'Gender', 'Color', 'Style']))

# Streamlit UI
ui_css = '''
<style>
.stApp > header {
    background-color: transparent;
}
.stApp {
    margin: auto;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: auto; 
    background-color: transparent;
    animation: gradient 15s ease infinite;
    background-size: 400% 400%;
    background-attachment: fixed;
}
</style>
'''
st.set_page_config(page_title='Fashion', page_icon=':shirt:', layout='centered', initial_sidebar_state='expanded')
st.title('Fashion Advisor')
st.markdown(ui_css, unsafe_allow_html=True)

with st.sidebar:
    gender_input = st.selectbox("Gender:", gender_categories)  # Dynamically populated
    color_input = st.selectbox("Color:", color_categories)
    style_input = st.selectbox("Style:", style_categories)  # Dynamically populated
    item_input = st.selectbox("Item:", item_categories)  # Dynamically populated

if st.button("Predict Fashion Item"):
    try:
        # Prepare user input for prediction
        user_data = pd.DataFrame({
            'Item_Name': [item_input],
            'Gender': [gender_input],
            'Color': [color_input],
            'Style': [style_input]
        })

        # One-hot encode the user input
        user_encoded = encoder.transform(user_data)
        user_vector = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Item_Name', 'Gender', 'Color', 'Style']))

        # Compute cosine similarity between user input and all rows in the dataset (use one-hot encoded `X`)
        similarities = cosine_similarity(user_vector, X)

        # Get the top 3 most similar items
        top_indices = np.argsort(similarities[0])[-3:][::-1]
        top_items = data_imputed.iloc[top_indices]
        # Format the results in a human-friendly form
        recommendations = []
        
        user_style = user_data['Style'].iloc[0]
        user_gender = user_data['Gender'].iloc[0]
        
        for index, item in top_items.iterrows():
            rec = f"When going to an occasion which is {user_style}, {item['Item_Name']} with color {item['Color']}, for the style {item['Style']} can be worn with the following Matching items({data.iloc[index,7]}) having these respective colors ( {data.iloc[index,8]})) for gender {data.iloc[index,6]}."
            
            # Append recommendation based on style and gender match
            if item['Style'] == user_style and item['Gender'] == user_gender:
                recommendations.append(rec)
        
        # Check if any recommendations were made
        if recommendations:
            st.success(f"I advise you to dress in the following way:\n" + "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]))
        else:
            st.success("There are no matching recommendations.")
            
    except ValueError as e:
        st.error(str(e))