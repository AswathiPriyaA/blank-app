import os
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Custom styles for black background and red/purple text
st.markdown(
    """
    <style>
    /* Full-page black background */
    .main, .reportview-container, .stApp {
        background-color: #000000;
        color: #870808;
    }
    
    /* Title styling */
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #8710b3;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Subtitle styling */
    .sub-title {
        font-size: 20px;
        color: #d10f25; /* Red */
        text-align: center;
        margin-bottom: 30px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #ad0909 !important;
        color: #3da6a6 !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        padding: 12px 28px !important;
        border: none !important;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #7656b3 !important;
    }

    /* Image styling */
    .stImage img {
        border-radius: 12px;
        object-fit: cover;
        width: 100% !important;
        height: auto !important;
    }

    /* Container for displaying images in rows of 3 */
    .img-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
    }

    /* Error Message Styling */
    .stAlert {
        color: #b19cd9 !important;
        background-color: #222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit Title
st.markdown('<div class="main-title">ACHU"S FASHIONS 👗 👢 🎀 🧣</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Find similar fashion images using ResNet50 and Cosine Similarity</div>', unsafe_allow_html=True)

# User input for search term and number of images
search_term = st.text_input("Enter search term for fashion images:", "fashion")
num_images = st.number_input("Number of images to fetch:", min_value=1, max_value=20, value=6)

# Load ResNet50 model
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# Function to fetch images from Pexels
def fetch_images_from_pexels(query, num_images):
    if not PEXELS_API_KEY:
        st.error("❌ Missing Pexels API Key. Please check your .env file.")
        return []
    
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"❌ Failed to fetch image URLs from Pexels (Status code: {response.status_code})")
        return []
    
    data = response.json()
    return [photo["src"]["medium"] for photo in data.get("photos", [])]

# Function to load and preprocess images
def load_and_preprocess_images(image_urls):
    images = []
    file_names = []
    
    for url in image_urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            file_names.append(url)
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
    
    if len(images) == 0:
        st.error("❌ No valid images found. Cannot proceed with prediction.")
        return None, None
    
    images = np.vstack(images)
    return images, file_names

# Function to find similar images using Cosine Similarity
def find_similar_images(features):
    similarity = cosine_similarity(features)
    similar_indices = similarity.argsort(axis=1)[:, ::-1]
    return similar_indices

# Main logic
if st.button("Find Similar Fashion Images"):
    st.write(f"🔎 Fetching '{search_term}' images from Pexels...")

    image_urls = fetch_images_from_pexels(search_term, num_images)

    if not image_urls:
        st.error("❌ No images found. Please try again later.")
    else:
        images, file_names = load_and_preprocess_images(image_urls)
        
        if images is not None:
            st.write("### 👗 Fetched Images:")
            rows = len(file_names) // 3 + (len(file_names) % 3 != 0)
            for row in range(rows):
                row_images = file_names[row * 3: (row + 1) * 3]
                cols = st.columns(3)
                for i, img_url in enumerate(row_images):
                    cols[i].image(img_url, caption=f"Image {row * 3 + i + 1}", width=100, use_container_width=False)

            # Extract features using ResNet50
            features = model.predict(images)

            # Find similar images
            similar_indices = find_similar_images(features)

            st.write("### 🌟 Similar Images:")
            for i in range(len(file_names)):
                st.write(f"**Base Image {i + 1}:**")
                st.image(file_names[i], width=100)
                
                similar_images = [file_names[idx] for idx in similar_indices[i][1:4]]
                
                cols = st.columns(3)
                for j, img_url in enumerate(similar_images):
                    cols[j].image(img_url, caption=f"Similar {j + 1}", width=100, use_container_width=False)
