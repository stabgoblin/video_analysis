import cv2
from captioner_lite import CaptionerLite
from activity_analyzer import ActivityAnalyzer
from PIL import Image
import os

def process_video(video_path, captioner, analyzer):
    """Generate and print one caption per second from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    second = 0

    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print(f"Video FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:  # Process one frame per second
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = captioner.generate_caption(image_pil)
            analyzer.update_log(caption)

            print(f"Second {second:>3}: {caption}")
            second += 1

        frame_count += 1

    cap.release()
    print("\nVideo analysis complete.")

def main():
    # Define your video path here
    video_path = "clips/clip_69.mp4"

    if not os.path.isfile(video_path):
        print(f"Error: File not found - {video_path}")
        return

    captioner = CaptionerLite()
    analyzer = ActivityAnalyzer()

    process_video(video_path, captioner, analyzer)

if __name__ == "__main__":
    main()
