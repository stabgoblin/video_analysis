import os
import cv2
from blip2_captioner import BLIP2Captioner
from activity_analyzer import ActivityAnalyzer

def process_clip(clip_path, captioner, analyzer):
    """Process a single video clip frame by frame"""
    cap = cv2.VideoCapture(clip_path)
    clip_name = os.path.basename(clip_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily for captioning
        frame_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

        # Generate caption and analyze
        caption = captioner.generate_caption(frame_path)
        analyzer.update_log(caption)

        # Check for alerts
        alerts = analyzer.check_alerts()
        if alerts:
            print(f"\nALERT in {clip_name} (Frame {frame_count}):")
            print(f"Caption: {caption}")
            print("Alerts:")
            print("\n".join(alerts))

        # Clean up temp file
        os.remove(frame_path)
        frame_count += 1

    cap.release()

def main():
    # Initialize components (keep your existing initialization)
    captioner = BLIP2Captioner()
    analyzer = ActivityAnalyzer()

    # Process all clips in the clips directory
    clips_dir = "clips"
    if not os.path.exists(clips_dir):
        print(f"Error: Directory '{clips_dir}' not found")
        return

    print(f"Analyzing clips in '{clips_dir}'...")
    for clip_name in sorted(os.listdir(clips_dir)):
        if clip_name.endswith(('.mp4', '.avi', '.mov')):  # Supported video formats
            clip_path = os.path.join(clips_dir, clip_name)
            print(f"\nProcessing clip: {clip_name}")
            process_clip(clip_path, captioner, analyzer)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()