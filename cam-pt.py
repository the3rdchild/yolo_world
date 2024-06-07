import cv2
import torch
import time
from ultralytics import YOLO
from threading import Thread
from queue import Queue

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Change to 'yolov8s.pt' for a slightly larger model
model.to('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, image, captions, box_threshold, device):
    results = model(image, conf=box_threshold)
    predictions = results[0]

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    for pred in predictions.boxes:
        box = pred.xyxy.cpu().numpy().astype(int)[0]
        score = pred.conf.cpu().numpy()[0]
        label = model.names[int(pred.cls.cpu().numpy()[0])]

        if label in captions and score >= box_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)

    return filtered_boxes, filtered_scores, filtered_labels

def webcam_reader(frame_queue):
    cap = cv2.VideoCapture(2)  # Open the default webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)

def process_webcam(captions, box_threshold, device):
    frame_queue = Queue(maxsize=10)
    reader_thread = Thread(target=webcam_reader, args=(frame_queue,))
    reader_thread.start()
    
    prev_frame_time = 0

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        
        # Resize the frame
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        
        # Predictions
        boxes, logits, labels = predict(
            model=model, 
            image=image_rgb, 
            captions=captions, 
            box_threshold=box_threshold, 
            device=device
        )
        
        # Annotate the frame
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Webcam', frame_resized)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cv2.destroyAllWindows()
    reader_thread.join()

# Set the text prompts
TEXT_PROMPTS = ["person, cell phone, tv, laptop"]  # Input prompt >= 1 ["dog, person, car, etc."]
BOX_THRESHOLD = 0.41  # Increase the confidence if the box detector is more than the real object count

# Process the webcam feed
process_webcam(TEXT_PROMPTS, BOX_THRESHOLD, 'cuda' if torch.cuda.is_available() else 'cpu')
