from ultralytics import YOLO
import cv2
import math 
import pyttsx3
import smtplib
from email.mime.text import MIMEText

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

text_speech = pyttsx3.init()

# Define the sender and recipient email addresses
sender_email = "felixpremnath.mca2023@adhiyamaan.in"
recipient_email = "felixprem02@gmail.com"

# Define the email subject and body
subject = "Person Detected"
body = "A person has been detected by the system."

# Create a text message
msg = MIMEText(body)
msg['Subject'] = subject
msg['From'] = sender_email
msg['To'] = recipient_email

# Set up the SMTP server
server = smtplib.SMTP('smtp.gmail.com', 123)  # Replace with your SMTP server and port
server.starttls()  # Use TLS encryption
server.login(sender_email, "gmail@2002")  # Replace with your email password

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes

        for box in boxes:
        
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            if(classNames[cls]=="person"):
                print("Danger",classNames[cls])
                text_speech.say("danger")
                text_speech.say(f"{classNames[cls]} danger detected with confidence {confidence}")
                text_speech.runAndWait()
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                # Send the email
                server.sendmail(sender_email, recipient_email, msg.as_string())
                print("Email sent successfully!")
              
    

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

server.quit()
cap.release()
cv2.destroyAllWindows()
