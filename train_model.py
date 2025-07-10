import cv2
import numpy as np
import os
import pickle
from PIL import Image
import time

# Konfigurasi
DATASET_DIR = "dataset"
MODEL_DIR = "training_data"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_FILE = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABEL_FILE = os.path.join(MODEL_DIR, "labels.pickle")

def prepare_training_data():
    """Mempersiapkan data training dari folder dataset"""
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    print("\n[INFO] Memproses dataset...")
    start_time = time.time()

    # Iterasi melalui setiap subfolder di dataset
    for root, dirs, files in os.walk(DATASET_DIR):
        for dir_name in dirs:
            if dir_name not in label_ids:
                label_ids[dir_name] = current_id
                current_id += 1
            
            user_dir = os.path.join(root, dir_name)
            print(f"- Memproses: {dir_name} (ID: {label_ids[dir_name]})")

            # Proses setiap gambar wajah
            for filename in os.listdir(user_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(user_dir, filename)
                    
                    # Konversi ke grayscale
                    pil_image = Image.open(image_path).convert('L')
                    image_np = np.array(pil_image, 'uint8')
                    
                    # Deteksi wajah (untuk memastikan kualitas data)
                    detected_faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)
                    
                    if len(detected_faces) == 1:
                        (x, y, w, h) = detected_faces[0]
                        face_roi = image_np[y:y+h, x:x+w]
                        faces.append(face_roi)
                        labels.append(label_ids[dir_name])

    # Simpan mapping label
    with open(LABEL_FILE, 'wb') as f:
        pickle.dump(label_ids, f)

    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Data selesai diproses dalam {elapsed_time:.1f} detik")
    print(f"- Total sample: {len(faces)}")
    print(f"- Jumlah user: {len(label_ids)}")
    
    return faces, labels, label_ids

def train_model():
    """Melatih model pengenalan wajah"""
    # Buat folder training jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Persiapan data
    faces, labels, label_ids = prepare_training_data()
    
    if len(faces) == 0:
        print("\n[ERROR] Tidak ada data training yang valid!")
        return

    print("\n[INFO] Melatih model...")
    start_time = time.time()
    
    # Gunakan LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Simpan model
    recognizer.save(MODEL_FILE)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Model berhasil dilatih dalam {elapsed_time:.1f} detik")
    print(f"- Model disimpan: {MODEL_FILE}")
    print(f"- Label mapping: {LABEL_FILE}")
    
    # Tampilkan summary
    print("\n=== Summary ===")
    for name, id in label_ids.items():
        count = labels.count(id)
        print(f"- {name} (ID:{id}): {count} samples")

if __name__ == "__main__":
    train_model()