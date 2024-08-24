Face Recognition Attendance System
This project implements a face recognition-based attendance system using Python, OpenCV, face_recognition library, and Tkinter for GUI.

Introduction
This project aims to automate attendance marking using facial recognition technology. It detects faces in real-time through a webcam feed, compares them against a database of known faces, records attendance with timestamps into a PostgreSQL database, and provides a user-friendly interface for interaction.

Features
Real-time face detection and recognition.
Automatic attendance recording with date and time stamps.
GUI application using Tkinter for login and attendance display.
PostgreSQL database integration for persistent storage of attendance records.
Requirements
Ensure you have the following installed before running the project:

Python 3.12
opencv-python 4.10.0
face-recognition 1.3.0
psycopg2 2.2
Pillow 10.4
Installation
Follow these steps to set up and run the project:

Clone the repository:

git clone https://github.com/your_username/face-recognition-attendance.git
cd face-recognition-attendance
Install dependencies:

pip install -r requirements.txt
Configure database:

Update database credentials (dbname, user, password, host) in main.py under SimpleFacerec.__init__.
Usage
To start the face recognition attendance system:

Run main.py:

python main.py
The Tkinter GUI window will open with live video feed.

Click on the "Login" button to start detecting faces and mark attendance.

Detected faces will be recognized and marked as present in the database.

Close the application by pressing 'q' or via the GUI.

How to Contribute
Contributions are welcome! Follow these steps to contribute:

Fork the repository and create a new branch.
Make your changes and test thoroughly.
Submit a pull request with a clear description of your additions or modifications.
Acknowledgements
Libraries and Tools
OpenCV: Used for video capture and basic image processing in Python.
face_recognition: Provided facial recognition capabilities, enhancing the accuracy of face detection and identification.
PostgreSQL: Used for storing attendance records securely and efficiently.
Pillow: Helped in image processing tasks within the Tkinter GUI.
