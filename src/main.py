import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi
IMG_SIZE = (100, 100)  # Ukuran gambar yang dinormalisasi
NUM_EIGENFACES = 10    # Jumlah eigenfaces yang akan dihitung

def load_images_from_folder(folder_path):
    """Memuat gambar dari folder dataset dan mengembalikan matriks gambar + labels"""
    image_vectors = []
    labels = []
    
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
    
    return np.array(image_vectors).T, np.array(labels)

def compute_mean_face(data_matrix):
    """Menghitung mean face dan matriks terpusat"""
    mean_face = np.mean(data_matrix, axis=1, keepdims=True)
    centered_matrix = data_matrix - mean_face
    return mean_face, centered_matrix

def compute_covariance_trick(centered_matrix):
    """Menghitung matriks kovarians menggunakan 'trick' untuk efisiensi"""
    return centered_matrix.T @ centered_matrix

def power_iteration(A, num_iter=1000, tol=1e-6):
    """Algoritma Power Iteration untuk mencari eigenvector dominan"""
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
    """Menghitung k eigenfaces teratas"""
    L = compute_covariance_trick(centered_matrix)
    eigenvectors = []
    
    for _ in range(k):
        _, eigenvector = power_iteration(L)
        eigenface = centered_matrix @ eigenvector
        eigenface = eigenface / np.linalg.norm(eigenface)
        eigenvectors.append(eigenface)
        
        # Deflasi: Kurangi komponen yang sudah ditemukan
        L = L - (eigenvector @ eigenvector.T) * (eigenvector.T @ L @ eigenvector)
    
    return np.array(eigenvectors).T

def project_face(face_vector, mean_face, eigenfaces):
    """Memproyeksikan wajah ke ruang eigenfaces"""
    centered = face_vector - mean_face.flatten()
    weights = eigenfaces.T @ centered
    return weights

def show_face(vector, title="Face"):
    """Menampilkan wajah dalam bentuk gambar"""
    face_img = vector.reshape(IMG_SIZE)
    plt.imshow(face_img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 1. Load Dataset
    dataset_path = "dataset/105_classes_pins_dataset"
    data_matrix, labels = load_images_from_folder(dataset_path)
    print(f"Dataset shape: {data_matrix.shape} (Setiap kolom adalah satu gambar)")
    
    # 2. Hitung Mean Face dan Centered Matrix
    mean_face, centered_matrix = compute_mean_face(data_matrix)
    print(f"Mean face shape: {mean_face.shape}")
    show_face(mean_face, "Mean Face")
    
    # 3. Hitung Eigenfaces
    eigenfaces = compute_top_eigenfaces(centered_matrix, NUM_EIGENFACES)
    print(f"Eigenfaces shape: {eigenfaces.shape} (Setiap kolom adalah eigenface)")
    
    # Tampilkan beberapa eigenfaces teratas
    for i in range(min(3, NUM_EIGENFACES)):  # Tampilkan 3 eigenfaces pertama
        show_face(eigenfaces[:, i], f"Eigenface {i+1}")
    
    # 4. Proyeksikan semua wajah ke ruang eigenfaces
    projected_weights = []
    for i in range(data_matrix.shape[1]):
        face_vec = data_matrix[:, i]
        weights = project_face(face_vec, mean_face, eigenfaces)
        projected_weights.append(weights)
    
    projected_weights = np.array(projected_weights)
    print(f"Projected weights shape: {projected_weights.shape}")
    
    # 5. Contoh pengenalan wajah (menggunakan Euclidean Distance)
    test_face_idx = 0  # Coba wajah pertama sebagai contoh
    test_weights = projected_weights[test_face_idx]
    
    # Hitung jarak ke semua wajah lain
    distances = np.linalg.norm(projected_weights - test_weights, axis=1)
    closest_idx = np.argmin(distances)
    
    print(f"\nHasil Pengenalan Wajah:")
    print(f"- Wajah uji: {labels[test_face_idx]}")
    print(f"- Wajah terdekat: {labels[closest_idx]}")
    print(f"- Jarak: {distances[closest_idx]}")
    
    # Tampilkan wajah asli dan hasil pengenalan
    show_face(data_matrix[:, test_face_idx], f"Test Face: {labels[test_face_idx]}")
    show_face(data_matrix[:, closest_idx], f"Closest Match: {labels[closest_idx]}")

    import numpy as np

def manual_eigen(A, k=None):
    # A: matriks yang sudah dikurangi mean (X_centered)
    print("Menghitung covariance matrix...")
    cov_matrix = np.dot(A, A.T)  # bentuk m x m
    print("Menghitung eigenvector dan eigenvalue (manual via SVD)...")

    # SVD langsung
    U, S, V = np.linalg.svd(cov_matrix)
    
    if k is not None:
        U = U[:, :k]
        S = S[:k]

    eigenfaces = np.dot(A.T, U)
    # Normalisasi
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
    
    return eigenfaces.T  # baris: eigenface

def project_faces(faces_matrix, mean_face, eigenfaces):
    # Kurangi mean face
    faces_centered = faces_matrix - mean_face
    # Proyeksikan ke eigenface
    return np.dot(faces_centered, eigenfaces.T)  # hasil: num_faces x k

def recognize_face(test_img, mean_face, eigenfaces, projected_faces):
    # Flatten test image & preprocessing
    test_vector = test_img.flatten()
    test_vector = test_vector - mean_face
    test_projection = np.dot(test_vector, eigenfaces.T)

    # Hitung Euclidean distance ke semua wajah di database
    distances = np.linalg.norm(projected_faces - test_projection, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    
    return min_index, min_distance

THRESHOLD = 2000  # sesuaikan dengan eksperimen

idx, dist = recognize_face(test_img, mean_face, eigenfaces, projected_faces)
if dist < THRESHOLD:
    print(f"Wajah dikenali sebagai wajah ke-{idx} dengan jarak {dist}")
else:
    print("Tidak ditemukan wajah yang mirip.")
