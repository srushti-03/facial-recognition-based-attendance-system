import os
import cv2
import face_recognition
import numpy as np
import psycopg2
from datetime import date, datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps

class SimpleFacerec:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="attendance_db",  # Replace with your database name
            user="postgres",         # Replace with your database username
            password="srushti",      # Replace with your database password
            host="localhost"         # Replace with your database host
        )
        self.cur = self.conn.cursor()

        # Initialize the attendance table if it doesn't exist
        self.create_attendance_table()

        # Load known faces and names from dataset
        self.known_face_encodings = []
        self.known_face_names = []
        dataset_path = r"C:\Users\LENOVO\Desktop\ig drones\face-recognition\dataset"  # Replace with your dataset folder path
        self.load_known_faces(dataset_path)

        # Keep track of today's date for attendance recording check
        self.today_date = date.today()

        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")

        # Set window dimensions
        self.root.geometry("1200x800")

        # Create main frames
        self.left_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, bg="#f0f0f0")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.right_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, bg="#f0f0f0")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left frame (video feed and controls)
        self.canvas = tk.Canvas(self.left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        self.welcome_label = tk.Label(self.left_frame, text="", font=("Arial", 18), bg="#f0f0f0")
        self.welcome_label.pack(pady=10)

        self.image_label = tk.Label(self.left_frame, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        self.login_button = tk.Button(self.left_frame, text="Login", command=self.login, bg="#28a745", fg="white", font=("Arial", 24), width=15, height=2)
        self.login_button.pack(pady=10)

        # Right frame (attendance list)
        self.attendance_heading = tk.Label(self.right_frame, text="Present Today", font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.attendance_heading.pack(pady=10)

        self.attendance_listbox = tk.Listbox(self.right_frame, font=("Arial", 18), width=30, height=25)
        self.attendance_listbox.pack(pady=10)

        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(0)

        # Load today's attendance from the database
        self.update_attendance_listbox()

        # Start the video loop
        self.video_loop()

        # Start the Tkinter main loop
        self.root.mainloop()

    def create_attendance_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status VARCHAR(10) NOT NULL
        );
        """
        self.cur.execute(create_table_sql)
        self.conn.commit()

    def load_known_faces(self, dataset_path):
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    if os.path.isfile(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            encoding = face_recognition.face_encodings(image_rgb)
                            if len(encoding) > 0:
                                face_encoding_np = np.array(encoding[0], dtype=np.float64)
                                self.known_face_encodings.append(face_encoding_np)
                                self.known_face_names.append(person_name)

    def record_attendance(self, name, status):
        current_time = datetime.now().time()
        check_sql = "SELECT * FROM attendance WHERE name=%s AND date=%s;"
        self.cur.execute(check_sql, (name, self.today_date))
        result = self.cur.fetchone()

        if result is None:
            insert_sql = "INSERT INTO attendance (name, date, time, status) VALUES (%s, %s, %s, %s);"
            self.cur.execute(insert_sql, (name, self.today_date, current_time, status))
            self.conn.commit()
            self.update_attendance_listbox()
            return "Welcome"
        else:
            return "Already marked present"

    def detect_known_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        detected_names = []

        for (top, right, bottom, left), encoding in zip(face_locations, frame_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            if name != "Unknown":
                status = "Present"
                welcome_message = self.record_attendance(name, status)
                detected_names.append(name)
                self.welcome_label.config(text=f"{welcome_message} {name}")

                # Fetch the image of the person from the dataset and display it
                image_path = os.path.join(r"C:\Users\LENOVO\Desktop\ig drones\face-recognition\dataset", name)
                if os.path.isdir(image_path):
                    for filename in os.listdir(image_path):
                        image_path = os.path.join(image_path, filename)
                        if os.path.isfile(image_path):
                            person_image = Image.open(image_path)
                            person_image = ImageOps.fit(person_image, (150, 150), Image.LANCZOS)
                            person_image = ImageTk.PhotoImage(person_image)
                            self.image_label.config(image=person_image)
                            self.image_label.image = person_image
                            break

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return detected_names

    def update_attendance_listbox(self):
        self.attendance_listbox.delete(0, tk.END)
        fetch_sql = "SELECT name FROM attendance WHERE date=%s AND status='Present';"
        self.cur.execute(fetch_sql, (self.today_date,))
        results = self.cur.fetchall()

        for row in results:
            self.attendance_listbox.insert(tk.END, row[0])

    def login(self):
        ret, frame = self.cap.read()
        if ret:
            detected_names = self.detect_known_faces(frame)
            if detected_names:
                self.welcome_label.config(text=f"Welcome {detected_names[0]}")
            else:
                messagebox.showerror("Error", "No known face detected")

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.detect_known_faces(frame)
            frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
            self.canvas.image = imgtk
            self.root.after(10, self.video_loop)

    def __del__(self):
        self.cur.close()
        self.conn.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sfr = SimpleFacerec()
