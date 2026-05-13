import streamlit as st
import pickle
import tempfile
import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

# ==========================
# LOAD MODEL
# ==========================

model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# ==========================
# LOAD FEATURES
# ==========================

with open("features.pkl", "rb") as f:
    data = pickle.load(f)

features_db = data["features"]
paths_db = data["paths"]

# ==========================
# PREPROCESS
# ==========================

def preprocess_image(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_features(img_path):

    img = preprocess_image(img_path)

    features = model.predict(img, verbose=0)

    features = features.flatten()

    features = features / np.linalg.norm(features)

    return features

# ==========================
# SEARCH
# ==========================

def find_similar(query_path, top_k=5):

    query_features = extract_features(query_path)

    similarities = cosine_similarity(
        [query_features],
        features_db
    )[0]

    indices = np.argsort(similarities)[::-1][:top_k]

    return [
        (paths_db[i], similarities[i])
        for i in indices
    ]

# ==========================
# STREAMLIT UI
# ==========================

st.title("AI Image Similarity Search Engine")

uploaded = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    temp_file.write(uploaded.read())

    st.image(temp_file.name, caption="Query Image")

    results = find_similar(temp_file.name)

    st.subheader("Similar Images")

    cols = st.columns(5)

    for idx, (img_path, score) in enumerate(results):

        cols[idx].image(
            img_path,
            caption=f"Score: {score:.2f}"
        )