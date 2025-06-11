import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time

IMG_SIZE = (100, 100)
NUM_EIGENFACES = 10
THRESHOLD_DEFAULT = 2000

def load_training_images(folder_path):
    image_vectors, labels, raw_images = [], [], []
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, IMG_SIZE)
                img_vector = img_resized.flatten()
                image_vectors.append(img_vector)
                labels.append(person_name)
                raw_images.append(img_resized)
    return np.array(image_vectors).T, np.array(labels), raw_images

def compute_mean_face(data_matrix):
    mean_face = np.mean(data_matrix, axis=1, keepdims=True)
    centered_matrix = data_matrix - mean_face
    return mean_face, centered_matrix

def compute_covariance_trick(centered_matrix):
    return centered_matrix.T @ centered_matrix

def power_iteration(A, num_iter=1000, tol=1e-6):
    n = A.shape[0]
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)
    for _ in range(num_iter):
        Ab = A @ b
        b_new = Ab / np.linalg.norm(Ab)
        if np.linalg.norm(b - b_new) < tol:
            break
        b = b_new
    eigenvalue = b.T @ A @ b
    return eigenvalue, b

def compute_top_eigenfaces(centered_matrix, k=10):
    L = compute_covariance_trick(centered_matrix)
    eigenvectors = []
    for _ in range(k):
        _, eigenvector = power_iteration(L)
        eigenface = centered_matrix @ eigenvector
        eigenface = eigenface / np.linalg.norm(eigenface)
        eigenvectors.append(eigenface)
        L = L - (eigenvector @ eigenvector.T) * (eigenvector.T @ L @ eigenvector)
    return np.array(eigenvectors).T

def project_face(face_vector, mean_face, eigenfaces):
    centered = face_vector - mean_face.flatten()
    weights = eigenfaces.T @ centered
    return weights

def main():
    st.title("🧠 Face Recognition App (EigenFace)")

    dataset_path = st.text_input("📁 Dataset Folder Path", "dataset")
    uploaded_file = st.file_uploader("📤 Upload Gambar Test", type=["jpg", "jpeg", "png"])
    threshold = st.slider("🎚️ Threshold Kemiripan", 0, 10000, value=THRESHOLD_DEFAULT)

    if st.button("🚀 Eksekusi") and uploaded_file is not None:
        start_time = time.time()

        # Load gambar test (tidak dari dataset)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        test_img_resized = cv2.resize(test_img, IMG_SIZE)
        test_vector = test_img_resized.flatten()

        # Load data latih dari folder dataset
        data_matrix, labels, raw_images = load_training_images(dataset_path)
        if data_matrix.size == 0:
            st.error("❌ Dataset kosong atau tidak ditemukan.")
            return

        mean_face, centered_matrix = compute_mean_face(data_matrix)
        eigenfaces = compute_top_eigenfaces(centered_matrix, NUM_EIGENFACES)

        projected_weights = []
        for i in range(data_matrix.shape[1]):
            weights = project_face(data_matrix[:, i], mean_face, eigenfaces)
            projected_weights.append(weights)
        projected_weights = np.array(projected_weights)

        test_weights = project_face(test_vector, mean_face, eigenfaces)
        distances = np.linalg.norm(projected_weights - test_weights, axis=1)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]

        col1, col2 = st.columns(2)
        col1.image(test_img_resized, caption="Test Image", width=200)
        col2.image(raw_images[closest_idx], caption=f"Matched Image: {labels[closest_idx]}", width=200)

        st.markdown("### 🔎 Hasil")
        if closest_distance < threshold:
            st.success(f"Wajah dikenali sebagai: **{labels[closest_idx]}**")
        else:
            st.error("Tidak ditemukan wajah yang cocok.")
        st.write(f"📏 Jarak Euclidean: {closest_distance:.2f}")
        st.write(f"⏱️ Waktu Eksekusi: {time.time() - start_time:.2f} detik")

if __name__ == "__main__":
    main()
