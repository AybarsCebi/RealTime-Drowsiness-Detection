import cv2
import numpy as np
from tensorflow.keras.models import load_model

from arrange_data import IMG_SIZE
import winsound
import time
from collections import deque

information = [
    "Drowsiness Analysis Scenarios to be Tested:",
    "1- Detection of closed eyes (sleeping person) for a certain duration",
    "2- Analysis of prolonged squinting",
    "3- Detection of head dropping forward during sleep",
    "4- Detection of prolonged looking to the right or left"
]


model = load_model('eye_state_model.h5')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_eye(eye):
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = eye.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return eye

def warning():
    global warning_count
    warning_count+=1
    print("DROWSINESS DETECTION, ", warning_count)
    winsound.Beep(1000, 1000)




cap = cv2.VideoCapture(0)



video_writer = None
video_filename = "output_video.avi"  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
record_duration = 120  
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))


print(f"Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}")

if frame_width == 0 or frame_height == 0 or fps == 0:
    print("Invalid camera properties. Please check your camera.")
    exit()


if not cap.isOpened():
    print("Camera did not open. Check it please")
    exit()


blink_duration = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))
threshold = int(fps * 0.14)  

head_down_frames = 0
head_down_threshold = int(fps * 1.2) 

warning_count=0
start_time = time.time()
alert_message=""

record_start_time=time.time()

if not video_writer.isOpened():
    print("Error opening video file for writing.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1200, 900)) 

    
    if (time.time() - record_start_time) <= record_duration:
        try:
            video_writer.write(frame)
            print("Frame written to video.")
        except Exception as e:
            print(f"Error writing frame to video: {e}")
            break
    
    
    if (time.time() - record_start_time) <= 7:
        for i, line in enumerate(information):
            cv2.putText(frame, line, (30, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:  # Eğer yüz algılanmıyorsa (kafa düşüyorsa, sağa sola uzun dönüyorsa)
        head_down_frames += 1
        if head_down_frames > head_down_threshold:
            warning()
    else:
        head_down_frames = 0


    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_upper = roi_gray[:h//2, :] #dudak bölgesini algılamasını engelledik
        eyes = eye_cascade.detectMultiScale(roi_gray_upper)


        for (ex, ey, ew, eh) in eyes: #göz kapanıklığı takibi
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            try:
                eye_input = preprocess_eye(eye)
                prediction = model.predict(eye_input)
                label = 'Open' if np.argmax(prediction) == 0 else 'Closed'

                
                if np.argmax(prediction) == 1:
                    print("CLOSED EYE DETECTED")
                    blink_duration += 1
                else:
                    if blink_duration > threshold: 
                        warning()

                    blink_duration = 0
                

        
                color = (0, 255, 0) if label == 'Open' else (0, 0, 255)
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
                cv2.putText(frame, label, (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except:
                pass
    

    current_time = time.time()
    if current_time - start_time >= 20:
        if warning_count > 3:
            alert_message = "DROWSINESS ALERT, PLEASE REST"
        else:
            alert_message = ""
        warning_count = 0  
        start_time = current_time

    if alert_message:
        cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.putText(frame, f"Warnings: {warning_count}", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    
    cv2.imshow('Real-time Eye State Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if video_writer:
    video_writer.release()
    print(f"Video saved successfully to {video_filename}")


cap.release()
cv2.destroyAllWindows()


