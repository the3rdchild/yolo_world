import os
import cv2
import onnx
import onnxruntime as rt
import numpy as np
from threading import Thread
from queue import Queue

# Load COCO labels
def load_coco_labels(file_path):
    coco_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if ',' in line:
                label, index = line.split(',')
                coco_labels[label] = int(index)
            else:
                print(f"Skipping invalid line: {line}")
    return coco_labels

# Load the COCO labels from a file
COCO_LABELS_PATH = "COCO_LABELS.txt"
COCO_LABELS = load_coco_labels(COCO_LABELS_PATH)

# Load the ONNX model
session = rt.InferenceSession("ssd300_vgg16.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define the image transformation
def preprocess(image):
    # Resize the image to 300x300 and normalize
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(session, image, caption, box_threshold):
    # Preprocess the image and run inference
    image_preprocessed = preprocess(image)
    outputs = session.run(None, {input_name: image_preprocessed})

    # Extract the bounding boxes, scores, and labels
    boxes = outputs[0][0][:, :4]  # First output, first batch, first 4 elements are boxes
    scores = outputs[0][0][:, 4]  # Next element is the confidence score
    labels = outputs[0][0][:, 5].astype(int)  # Last element is the class label

    # Get the class label for the given text prompt
    class_label = COCO_LABELS.get(caption.lower())
    if class_label is None:
        raise ValueError(f"Class '{caption}' not found in COCO dataset labels.")

    # Filter by class label and score threshold
    indices = [i for i, label in enumerate(labels) if label == class_label and scores[i] >= box_threshold]
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    phrases = [caption] * len(indices)

    return filtered_boxes.astype(int), filtered_scores, phrases

def video_reader(video_path, frame_queue):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)  # Indicate the end of the video

def process_video(video_path, caption, box_threshold):
    frame_queue = Queue(maxsize=10)
    reader_thread = Thread(target=video_reader, args=(video_path, frame_queue))
    reader_thread.start()

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Resize the frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Get predictions
        boxes, logits, phrases = predict(
            session=session, 
            image=image_rgb, 
            caption=caption, 
            box_threshold=box_threshold
        )

        # Annotate the frame using OpenCV
        for box, phrase in zip(boxes, phrases):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, phrase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    reader_thread.join()

# Define paths
HOME = "/media/nops/disk2/Download/perkuliahan/yolo/YOLO_WORLD/"
VIDEO_NAME = "source/videos/crowd.mp4"
VIDEO_PATH = os.path.join(HOME, VIDEO_NAME)

# Set the text prompt
TEXT_PROMPT = "dog"
BOX_THRESHOLD = 0.5  # Increase the confidence threshold

# Process the video
process_video(VIDEO_PATH, TEXT_PROMPT, BOX_THRESHOLD)
