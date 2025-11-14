import cv2
import os

# --- Settings ---
DATASET_DIR = "dataset"
SAMPLES_TO_TAKE = 10
PADDING = 30 # <-- NEW: We'll add 30 pixels of padding
# ----------------

cap = cv2.VideoCapture(0) # Use your CAMERA_INDEX if not 0

# Ask for the student's name (this will be the folder name)
student_name = input("Enter the student's name (no spaces): ")
student_dir = os.path.join(DATASET_DIR, student_name)

# Create the directory if it doesn't exist
os.makedirs(student_dir, exist_ok=True)
print(f"Directory created: {student_dir}")
print("Look at the camera. Press 'k' to capture an image.")
print(f"We will take {SAMPLES_TO_TAKE} samples. Press 'q' to quit early.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sample_count = 0
while cap.isOpened() and sample_count < SAMPLES_TO_TAKE:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_h, frame_w = frame.shape[:2] # Get frame dimensions
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    
    # --- MODIFIED: We need to define x,y,w,h outside the key press
    #     so we can draw the rectangle correctly
    
    face_coords = None # Holds our padded coordinates
    
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # --- NEW: Calculate padded coordinates ---
        # Ensure we don't go out of the frame boundaries
        y_start = max(0, y - PADDING)
        y_end = min(frame_h, y + h + PADDING)
        x_start = max(0, x - PADDING)
        x_end = min(frame_w, x + w + PADDING)
        
        face_coords = (x_start, y_start, x_end, y_end) # Store coords
        
        # Draw the rectangle (using original x,y,w,h for a tighter box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_TO_TAKE}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Create Dataset - Press 'k' to capture, 'q' to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF

    # --- MODIFIED: Use the 'face_coords' we calculated
    if key == ord('k') and face_coords:
        # Save the PADDED face region
        (x1, y1, x2, y2) = face_coords
        face_img = frame[y1:y2, x1:x2]
        
        img_path = os.path.join(student_dir, f"sample_{sample_count + 1}.png")
        cv2.imwrite(img_path, face_img)
        print(f"Saved sample {sample_count + 1} to {img_path}")
        sample_count += 1
        
    elif key == ord('q'):
        break

print("Dataset creation finished.")
cap.release()
cv2.destroyAllWindows()