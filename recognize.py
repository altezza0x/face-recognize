import cv2
import pickle

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training_data/face_recognizer.yml")

with open("training_data/labels.pickle", 'rb') as f:
    labels = {v:k for k,v in pickle.load(f).items()}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi dengan parameter optimal
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(100, 100)
    )
    
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        
        # Prediksi dengan confidence
        label_id, confidence = recognizer.predict(face_resized)
        
        if confidence < 70:  # Threshold bisa disesuaikan
            name = labels.get(label_id, "Unknown")
            color = (0, 255, 0)  # Hijau jika dikenali
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Merah jika tidak dikenali
        
        cv2.putText(frame, f"{name} {100-confidence:.1f}%", (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()