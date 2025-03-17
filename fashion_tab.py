import streamlit as st
import numpy as np
from PIL import Image
import os
import glob

# Sample data (you can replace this with actual data)
recent_searches = ["Red Dress", "Leather Jacket", "Sneakers"]
fashion_items = {
    "Red Dress": "A stylish red dress perfect for evening parties.",
    "Leather Jacket": "Classic black leather jacket, ideal for any weather.",
    "Sneakers": "Comfortable and trendy white sneakers."
}

# Load sample images (replace with your actual image path)
image_dir = "sample_images"
image_paths = glob.glob(f"{image_dir}/*.jpg")

def show_similar_images(selected_item):
    st.subheader("Similar Images")
    # Display random similar images (replace with your recommendation system)
    for img_path in image_paths[:3]:
        img = Image.open(img_path)
        st.image(img, width=150)

def login():
    st.subheader("🔒 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":  # Replace with secure auth
            st.success("Logged in successfully!")
            return True
        else:
            st.error("Invalid credentials")
            return False
    return False

def show_fashion_tab():
    if login():
        st.title("👗 Fashion Hub")
        
        # Display Fashion Items
        st.subheader("🛍️ Fashion Collection")
        for item, desc in fashion_items.items():
            st.markdown(f"**{item}** – {desc}")

        # Recently Searched Section
        st.subheader("📌 Recently Searched")
        for search in recent_searches:
            st.markdown(f"- {search}")

        # Similar Images Section
        st.subheader("🔎 Similar Images")
        selected_item = st.selectbox("Select an item to see similar images", list(fashion_items.keys()))
        if st.button("Show Similar Images"):
            show_similar_images(selected_item)
