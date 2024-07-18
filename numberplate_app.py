import streamlit as st
import cv2
import math
import logging
from ultralytics import YOLO
import cvzone
import tempfile

def process_video(video_file):
    # Load the YOLOv8 model for number plate detection
    model = YOLO('license_plate_detector.pt')  # Replace with your model path

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize frame counter
    frame_counter = 0

    # Create an empty container to display the frames
    stframe = st.empty()

    # Create an empty container to display detection status
    ststatus = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame counter
        frame_counter += 1

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Process frame with YOLOv8 model
        results = model(frame)

        # Reset detection status for each frame
        number_plate_detected = False

        # Draw bounding boxes and labels on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                cls = int(box.cls[0])
                if confidence > 30:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{model.names[cls]} {confidence}%'
                    cvzone.putTextRect(frame, label, [x1, y1 - 10], scale=1, thickness=1)

                    # Update detection status based on class
                    if model.names[cls] == 'license_plate':
                        number_plate_detected = True

        # Display the frame in the Streamlit app
        stframe.image(frame, channels="BGR")

        # Update detection status in the sidebar
        if number_plate_detected:
            ststatus.success(f"Frame {frame_counter}: Number plate detected - Yes")
        else:
            ststatus.error(f"Frame {frame_counter}: Number plate detected - No")

        # Log frame processing
        logging.info(f"Processed frame {frame_counter}")

    cap.release()

def main():
    st.title("Number Plate Detection in Video")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # Process the uploaded video directly
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    main()
