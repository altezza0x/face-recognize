import tkinter as tk
from tkinter import ttk, messagebox
import os
import pygame
import pygame.camera
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
import time

class FaceRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")  # Ukuran window diperbesar
        
        # Initialize pygame
        pygame.init()
        pygame.camera.init()
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.header = ttk.Label(self.main_frame, text="Face Recognition System", style='Header.TLabel')
        self.header.pack(pady=(0, 20))
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Capture Tab
        self.capture_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.capture_tab, text='Capture Faces')
        self.setup_capture_tab()
        
        # Train Tab
        self.train_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.train_tab, text='Train Model')
        self.setup_train_tab()
        
        # Recognize Tab
        self.recognize_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.recognize_tab, text='Recognize Faces')
        self.setup_recognize_tab()
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Camera variables
        self.cam = None
        self.is_capturing = False
        self.is_recognizing = False
        self.recognizer = None
        self.labels = {}
        
        # Key states
        self.key_states = {'space': False, 'q': False, 'esc': False}
        
        # Bind keyboard events
        self.root.bind('<KeyPress>', self.key_press)
        self.root.bind('<KeyRelease>', self.key_release)
        
        # Load model if exists
        self.load_model()

    def key_press(self, event):
        if event.keysym == 'space':
            self.key_states['space'] = True
        elif event.keysym == 'q':
            self.key_states['q'] = True
        elif event.keysym == 'Escape':
            self.key_states['esc'] = True

    def key_release(self, event):
        if event.keysym == 'space':
            self.key_states['space'] = False
        elif event.keysym == 'q':
            self.key_states['q'] = False
        elif event.keysym == 'Escape':
            self.key_states['esc'] = False

    def setup_capture_tab(self):
        frame = ttk.Frame(self.capture_tab)
        frame.pack(pady=20)
        
        ttk.Label(frame, text="User ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.user_id_entry = ttk.Entry(frame, width=30)
        self.user_id_entry.grid(row=0, column=1, pady=5, padx=5)
        
        self.capture_btn = ttk.Button(frame, text="Start Capture", command=self.start_capture)
        self.capture_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Video display frame - Diperbesar
        self.video_frame = ttk.LabelFrame(self.capture_tab, text="Camera Preview")
        self.video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)  # Diperbesar
        
        ttk.Label(self.capture_tab, 
                 text="Instructions:\n1. Enter User ID\n2. Press 'Start Capture'\n3. Press SPACE to capture image\n4. Press ESC to stop",
                 justify=tk.LEFT).pack(pady=10)

    def setup_train_tab(self):
        frame = ttk.Frame(self.train_tab)
        frame.pack(pady=20)
        
        self.train_btn = ttk.Button(frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(pady=10)
        
        self.train_status = tk.Text(self.train_tab, height=15, width=90, state=tk.DISABLED)  # Diperbesar
        self.train_status.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.train_status)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_status.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.train_status.yview)
        
        ttk.Label(self.train_tab, 
                 text="Instructions:\n1. Make sure you have captured face samples\n2. Press 'Train Model' to start training",
                 justify=tk.LEFT).pack(pady=10)

    def setup_recognize_tab(self):
        frame = ttk.Frame(self.recognize_tab)
        frame.pack(pady=20)
        
        self.recognize_btn = ttk.Button(frame, text="Start Recognition", command=self.start_recognition)
        self.recognize_btn.pack(pady=10)
        
        # Video display frame - Diperbesar
        self.recog_video_frame = ttk.LabelFrame(self.recognize_tab, text="Face Recognition")
        self.recog_video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.recog_video_label = tk.Label(self.recog_video_frame)
        self.recog_video_label.pack(expand=True, fill=tk.BOTH)  # Diperbesar
        
        ttk.Label(self.recognize_tab, 
                 text="Instructions:\n1. Press 'Start Recognition' to begin\n2. Press 'q' to stop",
                 justify=tk.LEFT).pack(pady=10)

    def start_capture(self):
        user_id = self.user_id_entry.get().strip()
        if not user_id:
            messagebox.showerror("Error", "Please enter User ID first!")
            return

        try:
            cam_list = pygame.camera.list_cameras()
            if not cam_list:
                raise Exception("No camera found")
                
            # Gunakan resolusi yang lebih besar (640x480)
            self.cam = pygame.camera.Camera(cam_list[0], (640, 480))
            self.cam.start()
            
            self.is_capturing = True
            self.capture_btn.config(state=tk.DISABLED)
            self.status_var.set("Capture active - SPACE to capture, ESC to stop")
            
            self.update_capture()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to access camera: {str(e)}")

    def update_capture(self):
        if not self.is_capturing:
            return
            
        try:
            # Get image from pygame camera
            img = self.cam.get_image()
            
            # Convert Pygame surface to Tkinter PhotoImage
            img_str = pygame.image.tostring(img, 'RGB')
            pil_img = Image.frombytes('RGB', img.get_size(), img_str)
            
            # Resize untuk menyesuaikan dengan frame
            pil_img = pil_img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Check for space key to capture
            if self.key_states['space']:
                self.key_states['space'] = False  # Reset key state
                self.capture_face(pil_img)
            
            # Check for escape key to stop
            if self.key_states['esc']:
                self.key_states['esc'] = False  # Reset key state
                self.stop_capture()
                return
            
            # Continue updating
            self.root.after(10, self.update_capture)
        except Exception as e:
            print(f"Camera error: {str(e)}")
            self.stop_capture()

    def capture_face(self, pil_img):
        user_id = self.user_id_entry.get().strip()
        
        # Convert to OpenCV format
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            
            # Ensure dataset directory exists
            dataset_dir = os.path.abspath("dataset")
            user_dir = os.path.join(dataset_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save image
            timestamp = int(time.time() * 1000)
            filename = f"{user_id}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(user_dir, filename), face_img)
            
            messagebox.showinfo("Success", f"Face sample saved as {filename}")
        else:
            messagebox.showwarning("Warning", "Please make sure exactly 1 face is visible!")

    def stop_capture(self):
        self.is_capturing = False
        if hasattr(self, 'cam') and self.cam:
            try:
                self.cam.stop()
            except:
                pass
        self.capture_btn.config(state=tk.NORMAL)
        self.status_var.set("Capture stopped")

    def train_model(self):
        self.train_btn.config(state=tk.DISABLED)
        self.status_var.set("Training model...")
        self.update_train_status("[INFO] Starting model training...\n")
        
        try:
            # Run training in a separate thread
            import threading
            threading.Thread(target=self._train_model_thread, daemon=True).start()
        except Exception as e:
            self.update_train_status(f"[ERROR] Failed to start training: {str(e)}\n")
            self.train_btn.config(state=tk.NORMAL)
            self.status_var.set("Training failed")

    def _train_model_thread(self):
        try:
            # Prepare training data
            faces, labels, label_ids = self.prepare_training_data()
            
            if len(faces) == 0:
                raise Exception("No valid training data found!")
            
            # Train the model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            
            # Save the model
            os.makedirs("training_data", exist_ok=True)
            recognizer.save("training_data/face_recognizer.yml")
            
            # Save labels
            with open("training_data/labels.pickle", 'wb') as f:
                pickle.dump(label_ids, f)
            
            self.update_train_status("[SUCCESS] Model trained successfully!\n")
            messagebox.showinfo("Success", "Model training completed!")
            
            # Reload the model
            self.load_model()
        except Exception as e:
            self.update_train_status(f"[ERROR] Training failed: {str(e)}\n")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.status_var.set("Training completed")

    def prepare_training_data(self):
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir):
            return [], [], {}
            
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = []
        labels = []
        label_ids = {}
        current_id = 0

        self.update_train_status("[INFO] Processing dataset...\n")
        
        for root, dirs, files in os.walk(dataset_dir):
            for dir_name in dirs:
                if dir_name not in label_ids:
                    label_ids[dir_name] = current_id
                    current_id += 1
                
                user_dir = os.path.join(root, dir_name)
                self.update_train_status(f"- Processing: {dir_name} (ID: {label_ids[dir_name]})\n")

                for filename in os.listdir(user_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(user_dir, filename)
                        
                        try:
                            # Read and convert to grayscale
                            image = cv2.imread(image_path)
                            if image is None:
                                continue
                                
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Face detection
                            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                            
                            if len(detected_faces) == 1:
                                (x, y, w, h) = detected_faces[0]
                                face_roi = gray[y:y+h, x:x+w]
                                faces.append(face_roi)
                                labels.append(label_ids[dir_name])
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
                            continue

        self.update_train_status(f"[INFO] Dataset processed. Total samples: {len(faces)}, Users: {len(label_ids)}\n")
        return faces, labels, label_ids

    def load_model(self):
        try:
            if os.path.exists("training_data/face_recognizer.yml") and os.path.exists("training_data/labels.pickle"):
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read("training_data/face_recognizer.yml")
                
                with open("training_data/labels.pickle", 'rb') as f:
                    self.labels = {v:k for k,v in pickle.load(f).items()}
                
                self.update_train_status("[INFO] Model loaded successfully\n")
        except Exception as e:
            self.update_train_status(f"[WARNING] Failed to load model: {str(e)}\n")

    def update_train_status(self, message):
        self.train_status.config(state=tk.NORMAL)
        self.train_status.insert(tk.END, message)
        self.train_status.see(tk.END)
        self.train_status.config(state=tk.DISABLED)

    def start_recognition(self):
        if not self.recognizer or not self.labels:
            messagebox.showerror("Error", "Model not trained! Please train the model first.")
            return
        
        try:
            cam_list = pygame.camera.list_cameras()
            if not cam_list:
                raise Exception("No camera found")
                
            # Gunakan resolusi yang lebih besar (640x480)
            self.recog_cam = pygame.camera.Camera(cam_list[0], (640, 480))
            self.recog_cam.start()
            
            self.is_recognizing = True
            self.recognize_btn.config(state=tk.DISABLED)
            self.status_var.set("Recognition active - Press 'q' to stop")
            
            self.update_recognition()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition: {str(e)}")

    def update_recognition(self):
        if not self.is_recognizing:
            return
            
        try:
            # Get image from pygame camera
            img = self.recog_cam.get_image()
            
            # Convert to OpenCV format
            img_str = pygame.image.tostring(img, 'RGB')
            pil_img = Image.frombytes('RGB', img.get_size(), img_str)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(100, 100))
            
            # Recognition
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                
                # Predict
                label_id, confidence = self.recognizer.predict(face_resized)
                
                if confidence < 70:  # Confidence threshold
                    name = self.labels.get(label_id, "Unknown")
                    color = (0, 255, 0)  # Green
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle and label
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(cv_img, f"{name} {100-confidence:.1f}%", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Convert back to Tkinter format dan resize
            result_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            result_img = result_img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=result_img)
            
            self.recog_video_label.imgtk = imgtk
            self.recog_video_label.configure(image=imgtk)
            
            # Check for quit command
            if self.key_states['q']:
                self.key_states['q'] = False
                self.stop_recognition()
                return
            
            # Continue updating
            self.root.after(10, self.update_recognition)
        except Exception as e:
            print(f"Recognition error: {str(e)}")
            self.stop_recognition()

    def stop_recognition(self):
        self.is_recognizing = False
        if hasattr(self, 'recog_cam') and self.recog_cam:
            try:
                self.recog_cam.stop()
            except:
                pass
        self.recognize_btn.config(state=tk.NORMAL)
        self.status_var.set("Recognition stopped")
        
        # Clear the video label
        self.recog_video_label.configure(image='')

    def on_closing(self):
        self.stop_capture()
        self.stop_recognition()
        try:
            pygame.quit()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()