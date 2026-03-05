from flask import Flask, render_template, Response
import cv2
import winsound
from ultralytics import YOLO
import threading
from playsound import playsound

app = Flask(__name__)

# load YOLO model
model = YOLO("yolov8n.pt")

# open camera
camera = cv2.VideoCapture(0)

alert_playing = False

# alert sound function
def play_alert():
    global alert_playing
    try:
        winsound.PlaySound("static/alert.wav", winsound.SND_FILENAME)
    except Exception as e:
        print("Sound error:", e)
    alert_playing = False

def generate_frames():
    global alert_playing

    while True:

        success, frame = camera.read()

        if not success:
            break

        # detect objects
        results = model(frame)

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])
                name = model.names[cls]

                if name == "cell phone":

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    print("PHONE DETECTED")  # debug

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

                    cv2.putText(frame,"PHONE DETECTED",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,0,255),
                                2)

                    if not alert_playing:
                        alert_playing = True
                        threading.Thread(target=play_alert).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)