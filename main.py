import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

# ==============================
# LOAD PRETRAINED MODEL
# ==============================

model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# ==============================
# IMAGE PREPROCESSING
# ==============================

def preprocess_image(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(img_path):

    processed_img = preprocess_image(img_path)

    features = model.predict(processed_img, verbose=0)

    flattened = features.flatten()

    normalized = flattened / np.linalg.norm(flattened)

    return normalized


# ==============================
# BUILD FEATURE DATABASE
# ==============================

dataset_path = "dataset"

image_paths = []
feature_list = []

print("Extracting Features...\n")

for file in os.listdir(dataset_path):

    full_path = os.path.join(dataset_path, file)

    try:
        features = extract_features(full_path)

        feature_list.append(features)

        image_paths.append(full_path)

        print(f"Processed: {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

feature_list = np.array(feature_list)

# ==============================
# SAVE FEATURES
# ==============================

with open("features.pkl", "wb") as f:
    pickle.dump({
        "features": feature_list,
        "paths": image_paths
    }, f)

print("\nFeatures Saved Successfully!")

# ==============================
# LOAD FEATURES
# ==============================

with open("features.pkl", "rb") as f:
    data = pickle.load(f)

features_db = data["features"]
paths_db = data["paths"]

# ==============================
# FIND SIMILAR IMAGES
# ==============================

def find_similar_images(query_img, top_k=5):

    query_features = extract_features(query_img)

    similarities = cosine_similarity(
        [query_features],
        features_db
    )[0]

    indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in indices:

        results.append({
            "image": paths_db[idx],
            "score": similarities[idx]
        })

    return results


# ==============================
# TEST SEARCH
# ==============================

query_image = "query.jpg"

results = find_similar_images(query_image, top_k=5)

print("\nTop Similar Images:\n")

for result in results:
    print(result["image"], " -> ", round(result["score"], 4))


# ==============================
# DISPLAY RESULTS
# ==============================

plt.figure(figsize=(15, 5))

# Query Image
query = cv2.imread(query_image)
query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

plt.subplot(1, 6, 1)
plt.imshow(query)
plt.title("Query")
plt.axis("off")

# Similar Images
for i, result in enumerate(results):

    img = cv2.imread(result["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 6, i + 2)
    plt.imshow(img)

    plt.title(
        f"{round(result['score'], 2)}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()