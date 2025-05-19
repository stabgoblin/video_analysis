import cv2
import os
from datetime import datetime

def extract_frames(video_path, output_dir="frames", fps=1):
    """Extract frames at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{output_dir}/frame_{timestamp}_{frame_count}.jpg", frame)
        
        frame_count += 1
    cap.release()

def enhance_night_image(image_path):
    """CLAHE enhancement for low-light frames."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    cv2.imwrite(image_path, enhanced)  # Overwrite with enhanced image