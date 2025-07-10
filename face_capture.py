import cv2
import os
import time

# Konfigurasi
DATASET_DIR = os.path.abspath("dataset")
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MAX_SAMPLES = 50
IMAGE_SIZE = (200, 200)

def ensure_dir():
    """Pastikan folder dataset ada dan bisa ditulis"""
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        # Test write permission
        test_file = os.path.join(DATASET_DIR, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"Error akses folder: {str(e)}")
        return False

def capture_faces():
    if not ensure_dir():
        return

    user_id = input("Masukkan ID User: ").strip()
    user_dir = os.path.join(DATASET_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa diakses!")
        return

    sample_count = 0
    print("\nTekan SPASI untuk ambil gambar, ESC untuk berhenti")

    while sample_count < MAX_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        # Debug: Tampilkan jumlah wajah terdeteksi
        debug_text = f"Detected: {len(faces)} | Samples: {sample_count}/{MAX_SAMPLES}"
        cv2.putText(frame, debug_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow(f"Face Capture - {user_id}", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPASI
            if len(faces) == 1:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, IMAGE_SIZE)
                
                timestamp = int(time.time() * 1000)
                filename = f"{user_id}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(user_dir, filename), face_img)
                sample_count += 1
                print(f"Sample {sample_count} tersimpan: {filename}")
            else:
                print("Harap pastikan hanya 1 wajah yang terlihat!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProses selesai. Sample tersimpan di: {user_dir}")

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
    print("=== PROGRAM PENGAMBILAN SAMPLE WAJAH ===")
    print(f"Lokasi dataset: {DATASET_DIR}")
    capture_faces()