import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

# Files for attendance and daily summary
attendance_file = 'attendance.xlsx'

# Ensure directories exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Function to cleanup OpenCV resources
def cleanup_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

# Function to register a new user
def register_user():
    name = simpledialog.askstring("Input", "Enter your name:")
    if not name:
        return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access the camera!")
        return

    try:
        while count < 30:  # Capture 30 images
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from the camera!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imwrite(os.path.join('dataset', f"{name}_{count}.jpg"), gray[y:y + h, x:x + w])
                count += 1

            cv2.imshow('Registering User', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cleanup_camera(cap)

    if count == 30:
        messagebox.showinfo("Success", f"User {name} registered successfully!")
    else:
        messagebox.showerror("Error", "Registration incomplete. Try again!")

# Function to train the recognizer
def train_recognizer():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        messagebox.showerror("Error", "LBPHFaceRecognizer not found. Ensure 'opencv-contrib-python' is installed.")
        return

    image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]
    faces = []
    ids = []
    names = {}

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = len(names)
        name = os.path.basename(image_path).split('_')[0]
        if name not in names:
            names[name] = label
        faces.append(img)
        ids.append(names[name])

    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    with open('names.txt', 'w') as f:
        f.write(str(names))
    messagebox.showinfo("Success", "Model trained successfully!")

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        df = pd.read_excel(attendance_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

    # Check if the student is already marked for today's date
    if not df.empty and ((df['Name'] == name) & (df['Date'] == date_str)).any():
        messagebox.showinfo("Attendance", f"Attendance for {name} already marked today.")
        return

    new_record = {'Name': name, 'Date': date_str, 'Time': time_str}
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    
    try:
        df.to_excel(attendance_file, index=False)
        messagebox.showinfo("Attendance", f"Attendance marked for {name} at {time_str}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save attendance: {e}")

# Function to take attendance
def take_attendance():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        messagebox.showerror("Error", "LBPHFaceRecognizer not found. Ensure 'opencv-contrib-python' is installed.")
        return

    if not os.path.exists('trainer.yml'):
        messagebox.showerror("Error", "No trained model found. Train the recognizer first!")
        return

    recognizer.read('trainer.yml')

    with open('names.txt', 'r') as f:
        names = eval(f.read())

    recognized_name = None  # Variable to store the recognized name

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from the camera!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id_, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 50:  # Confidence threshold
                    recognized_name = [k for k, v in names.items() if v == id_][0]
                    cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow('Taking Attendance', frame)

            # Break the loop once a person is recognized
            if recognized_name:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cleanup_camera(cap)

    if recognized_name:
        mark_attendance(recognized_name)

# Tkinter GUI
root = tk.Tk()
root.title("Facial Recognition Attendance System")

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        cv2.destroyAllWindows()

root.protocol("WM_DELETE_WINDOW", on_closing)

register_button = tk.Button(root, text="Register User", command=register_user)
register_button.pack(pady=10)

train_button = tk.Button(root, text="Train Recognizer", command=train_recognizer)
train_button.pack(pady=10)

take_attendance_button = tk.Button(root, text="Take Attendance", command=take_attendance)
take_attendance_button.pack(pady=10)

root.mainloop()
