from ultralytics import YOLO

# load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# class names
classNames = model.names

def detect_phone(frame):

    results = model(frame, stream=True)

    phone_detected = False

    for r in results:-
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            classname = classNames[cls]

            if classname == "cell phone":
                phone_detected = True

    return phone_detected, results