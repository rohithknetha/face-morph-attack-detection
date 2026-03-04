import cv2
import os

# Define paths for input and output
BASE_PATH = "F:/Major Project/Datasets"
REAL_PATH = os.path.join(BASE_PATH, "REAL")
FAKE_PATH = os.path.join(BASE_PATH, "FAKE")
OUTPUT_PATH = os.path.join(BASE_PATH, "Extracted_Frames")

# Create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_PATH, "REAL"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "FAKE"), exist_ok=True)

# Function to extract frames with reduced frequency
def extract_frames(video_path, output_folder, label, skip_frames=20):
    for folder in os.listdir(video_path):  
        folder_path = os.path.join(video_path, folder)

        if os.path.isdir(folder_path):
            for video_file in os.listdir(folder_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_file_path = os.path.join(folder_path, video_file)
                    
                    # Skip if frames already extracted
                    video_name = os.path.splitext(video_file)[0]
                    output_video_folder = os.path.join(output_folder, label, video_name)
                    if os.path.exists(output_video_folder) and len(os.listdir(output_video_folder)) > 0:
                        print(f"✅ Skipping {video_file} (already processed)")
                        continue

                    os.makedirs(output_video_folder, exist_ok=True)

                    cap = cv2.VideoCapture(video_file_path)
                    frame_count = 0
                    saved_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break  

                        if frame_count % skip_frames == 0:
                            resized_frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
                            frame_filename = f"frame_{saved_count:04d}.jpg"
                            cv2.imwrite(os.path.join(output_video_folder, frame_filename), 
                                        resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Compressed JPG
                            saved_count += 1
                        
                        frame_count += 1
                    
                    cap.release()
                    print(f"Extracted {saved_count} frames from {video_file}")

# Extract frames with reduced frequency
extract_frames(REAL_PATH, OUTPUT_PATH, "REAL", skip_frames=40)
extract_frames(FAKE_PATH, OUTPUT_PATH, "FAKE", skip_frames=40)

print("🎯 Frame extraction complete!")
