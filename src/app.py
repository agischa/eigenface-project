import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time

# Load fungsi dari kode kamu (boleh langsung copy paste juga)

IMG_SIZE = (100, 100)
NUM_EIGENFACES = 10

st.title("👁️‍🗨️ Face Recognition App dengan EigenFace")

# Folder Dataset
dataset_path = st.text_input("📁 Dataset Folder Path", "")

# Upload Gambar Uji
uploaded_file = st.file_uploader("📤 Upload Gambar Uji", type=["jpg", "jpeg", "png"])

# Slider threshold
threshold = st.slider("🎯 Threshold Kemiripan", min_value=1000, max_value=5000, value=2000)

if st.button("🚀 Jalankan Pengenalan"):
    if dataset_path and uploaded_file:
        start_time = time.time()

        # --- Proses Pengenalan Wajah ---
        data_matrix, labels = load_images_from_folder(dataset_path)
        mean_face, centered_matrix = compute_mean_face(data_matrix)
        eigenfaces = compute_top_eigenfaces(centered_matrix, NUM_EIGENFACES)
        projected_faces = np.array([
            project_face(data_matrix[:, i], mean_face, eigenfaces)
            for i in range(data_matrix.shape[1])
        ])

        # Proses Gambar Test
        img = Image.open(uploaded_file).convert('L')
        img = img.resize(IMG_SIZE)
        test_img = np.array(img)

        # Lakukan pengenalan wajah
        test_vector = test_img.flatten()
        test_projection = project_face(test_vector, mean_face, eigenfaces)
        distances = np.linalg.norm(projected_faces - test_projection, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        exec_time = time.time() - start_time

        # Output
        st.image(img, caption="Gambar Uji", width=150)
        if min_distance < threshold:
            st.success(f"✅ Dikenali sebagai: {labels[min_index]} (jarak: {min_distance:.2f})")
            matched_path = os.path.join(dataset_path, labels[min_index])
            if os.path.isdir(matched_path):
                matched_img_name = os.listdir(matched_path)[0]
                matched_img = cv2.imread(os.path.join(matched_path, matched_img_name), cv2.IMREAD_GRAYSCALE)
                matched_img = cv2.resize(matched_img, IMG_SIZE)
                st.image(matched_img, caption=f"Gambar Cocok ({labels[min_index]})", width=150)
        else:
            st.warning("❌ Tidak ada kecocokan ditemukan.")

        st.write(f"⏱️ Waktu Eksekusi: {exec_time:.2f} detik")
    else:
        st.error("Mohon isi path dataset dan upload gambar uji.")
