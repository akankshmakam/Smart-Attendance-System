import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import mysql.connector

mydb = mysql.connector.connect(
    host='localhost', user='root', passwd='6768', port='3306', database='attendence'
)
mycursor = mydb.cursor()

today = datetime.now()
formatted_date = today.strftime('%Y-%m-%d %H:%M:%S')

path = 'faces'
images = []
classNames = []
registered_names = []

def get_register_number_and_name(filename):
    filename = os.path.splitext(filename)[0]
    parts = filename.split('_')
    if len(parts) == 2:
        register_number = parts[0]
        name = parts[1]
        return register_number, name
    else:
        return None, None

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    register_number, name = get_register_number_and_name(cl)
    if register_number is not None and name is not None:
        curimg = cv2.imread(f'{path}/{cl}')
        images.append(curimg)
        classNames.append(name)
        registered_names.append(register_number)
        print(f"Register Number: {register_number}, Name: {name}")

def encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnown = encodings(images)
print("Encoding Complete...")

cap = cv2.VideoCapture(0)

# Adjust the table and column names as per your database setup
table_name = "khacks"
column_register_number = "Register_Number"
column_name = "Name"
column_entry = "Entry"


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            register_number = registered_names[matchIndex]
            print(f"Register Number: {register_number}, Name: {name}")
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if name not in registered_names:


                # Insert the new entry into the database
                add_people = "INSERT INTO khacks (Register_Number, Name, Entry) VALUES (%s, %s, %s)"
                students = (register_number, name, formatted_date)
                mycursor.execute(add_people, students)
                mydb.commit()

                # Add the name to the list of registered names
                registered_names.append(name)

        else:
            print('error')

    cv2.imshow('Image', img)
    cv2.waitKey(1)
