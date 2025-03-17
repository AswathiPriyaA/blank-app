import streamlit as st
from PIL import Image
from model import get_recommendations

st.set_page_config(page_title="Fashion Recommendation System", layout="centered")

# Title
st.title("👗 Fashion Recommendation System")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    # Button to get recommendations
    if st.button('Get Recommendations'):
        st.write("🔍 Finding similar images...")
        
        # Get recommendations from the model
        recommendations = get_recommendations(image)

        st.write("### Recommended Fashion Items:")
        cols = st.columns(len(recommendations))

        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                st.image(rec, width=150)

