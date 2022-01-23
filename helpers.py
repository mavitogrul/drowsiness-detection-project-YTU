import os
from scipy.spatial import distance
from twilio.rest import Client


# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    # 2.0 * C değerinin ear oranını kamera ile daha hassas algılaması için
    return ear


# Creating the dataset
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


account_sid = 'ACa0e7cfbae23320c42cdf035119498461'
auth_token = 'bae25a76c0645db79df3588c377c124c'

client = Client(account_sid, auth_token)
