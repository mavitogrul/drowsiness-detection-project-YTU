import csv
import time

from imutils import face_utils
from pygame import mixer
import dlib
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import numpy as np
import helpers
from helpers import client
import pickle
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_template import html
import datetime as dt
import os
import pickle

template = html

mixer.init()
sound = mixer.Sound('alarm.MP3')

# EAR list which will be saved and drawn later
total_ear = []

# Counter list which also represent frame number or time
total_ts = []

# Threshold value is 0.25, generally (according to our tests) EAR value will be around 0.3-0.5 when eyes are open,
# for closed eyes EAR value will be 0.25-0.05, now we can also change the EAR check value according to the formula in
# function "eye_aspect_ratio", (ear = (A + B) / (2.0 * C)) here value will be very close to 0 if eyes are closed,
# the reason for being 2.0 * C in the denominator, we want the EAR value to be more sensitive. If we divide with C in
# the formula, it will also need to increase the "thresh" value to understand that the eye is closed, this will also
# cause the sensitivity to decrease.

# If too much sensitive, the camera can give erroneous results.
# If non-sensitive, gives late responses when the eyes are closed or opened.


# If EAR value get below thresh value, it means it's time to count 20 frame to ensure the driver is sleepy
thresh = 0.25

# Count 20 frame to alert the driver
frame_check = 20

# Get the dlib's frontal face detector
detect = dlib.get_frontal_face_detector()

# Dlib will use this database to find "faces" and "facial structures"
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start to record
cap = cv2.VideoCapture(0)

# Frame count value
flag = 0

# Saves taken "drowsiness photo" to the given folder
helpers.assure_path_exists("dataset/")
control = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with open('email_list.csv', 'w', newline='') as e:
    writer = csv.writer(e)
    email_list = [["isim", "email"],
                  ["Tolga Öcal", "tolgaocal80@gmail.com"],
                  ["Ahmet Faruk Sırma", "ahmetfaruk2557@gmail.com"],
                  ["Mavi Toğrul", "mavitogrul@gmail.com"]]
    writer.writerows(email_list)

def send_email(receivers, html):

    sender_email = "ytu.tez.deneme@gmail.com"
    password = "ytu.tez.deneme.2022"
    receiver_email = receivers
    subject = "Python Test"
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Turn these into plain/html MIMEText objects
    part2 = MIMEText(html, "html")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first

    filename = "op_webcam.csv"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)
    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    # Add attachment to message and convert message to string
    message.attach(part2)
    message.attach(part)
    text = message.as_string()
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

def send_email_at():
    if os.path.isfile('important'):
        print("kullanıcılar mevcut")
        dosya = open('important', 'rb')
        loaded_dict = pickle.load(dosya)
        dosya.close()

        for email in loaded_dict:
            if loaded_dict[email] + dt.timedelta(weeks=1) <= dt.datetime.now():
                print(f"Sending email to {email}")
                html = template.replace("user_name", email)
                send_email(email, html)
                print('email sent')
                data = {email: dt.datetime.now()}
                dosya = open('important', 'wb')
                pickle.dump(data, dosya)
                dosya.close()
            else:
                print("Email gönderilmedi, süresi gelmedi")
    else:
        print("kullanıcılar mevcut değil, yeni kullanıcı ekleniyor")
        with open("email_list.csv") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            data = {}
            for isim, email in reader:
                print(f"Sending email to {isim} : email : {email}")
                html = template.replace("user_name", isim)
                send_email(email, html)
                data[email] = dt.datetime.now()

            dosya2 = open('important', 'wb')
            pickle.dump(data, dosya2)
            dosya2.close()

while cap.isOpened():

    send_email_at()

    # Extract a frame
    # ret, frame = cap.read()
    # Resize the frame
    # frame = imutils.resize(frame, width=600)

    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance
    image.flags.writeable = False
    # Get the result
    results = face_mesh.process(image)
    # To improve performance
    image.flags.writeable = True
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array

            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 720
            y = angles[1] * 720
            z = angles[2] * 720

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            subjects = detect(gray, 0)

            cv2.putText(image, "PRESS 'Q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            # Now loop over all the face detections and apply the predictor
            for subject in subjects:

                shape = predict(gray, subject)

                # Convert it to a (68, 2) size numpy array
                shape = face_utils.shape_to_np(shape)  # converting to NumPy Array

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                # Compute the EAR for both the eyes
                leftEAR = helpers.eye_aspect_ratio(leftEye)
                rightEAR = helpers.eye_aspect_ratio(rightEye)

                # Take the average of both the EAR
                ear = (leftEAR + rightEAR) / 2.0

                total_ear.append(ear)

                # Compute the convex hull for both the eyes and then visualize it
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

                # Draw a rectangle over the detected face
                (x, y, w, h) = face_utils.rect_to_bb(subject)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put a number
                cv2.putText(image, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the contours
                cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

                cv2.putText(image, "EAR: " + str(round(ear, 2)), (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if ear < thresh:
                    control = 0
                    flag += 1
                    cv2.drawContours(image, [leftEyeHull], -1, (0, 0, 255), 1)
                    cv2.drawContours(image, [rightEyeHull], -1, (0, 0, 255), 1)

                    if flag >= frame_check:

                        cv2.putText(image, "****************ALERT!****************", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(image, "****************ALERT!****************", (30, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if control == 0:
                            # Add the frame to the dataset ar a proof of drowsy driving
                            cv2.imwrite("dataset/frame_sleep%d.jpg" % control, image)
                            print("IF AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                            message = client.messages.create(
                                body='Sürücünün uykulu olduğu tespit edildi ! Dikkatli olun !',
                                from_='+16075369130',
                                to='+905314983563')

                        control = 1

                        try:
                            sound.play()
                        except:
                            pass
                else:
                    flag = 0
                    control = 0

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow("Head Pose Estimation", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 1) verileri bir dosyaya "tarih" ile birlikte kaydet, bu dosyayı her hafta email ile gönder
# 2) anlık "uyku" algılandığında "sms" gönder

for i in range(len(total_ear)):
    total_ts.append(i)

a = total_ear
c = total_ts

df = pd.DataFrame({"EAR": a, "TIME": c})
df.to_csv("op_webcam.csv", index=True)

# df = pd.read_csv("op_webcam.csv")
# df.plot(x='TIME', y='EAR')

plt.xticks(rotation=45, ha='right')
plt.title('EAR calculation over time of webcam')
plt.ylabel('EAR')
plt.plot(c, a)
plt.gcf().autofmt_xdate()
plt.show()
cv2.destroyAllWindows()

cap.release()
