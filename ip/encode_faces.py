import face_recognition
import pickle
import os

# --- Settings ---
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
# ----------------

print("Starting face encoding... This may take a moment.")

known_encodings = []
known_names = []

# Loop through each person in the training set
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    
    # Skip files
    if not os.path.isdir(person_dir):
        continue
        
    print(f"[+] Processing {person_name}...")
    
    sample_count = 0
    # Loop through each image for this person
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        
        try:
            # Load the image
            image = face_recognition.load_image_file(img_path)
            
            # Find face locations
            boxes = face_recognition.face_locations(image, model="hog")
            
            # --- MODIFICATION ---
            # Get all encodings for the image
            encodings_list = face_recognition.face_encodings(image, boxes)
            
            # Check if the list is NOT empty
            if encodings_list:
                # Get the first (and likely only) encoding
                encoding = encodings_list[0]
                
                # Add to our lists
                known_encodings.append(encoding)
                known_names.append(person_name)
                sample_count += 1
            else:
                # This was the error, but now it's just a warning
                print(f"[!] Warning: No face encoding found in {img_path}. Skipping.")
            # --- END OF MODIFICATION ---
            
        except Exception as e:
            print(f"[!] Warning: Could not process {img_path}. Skipping. (Error: {e})")

    print(f"[+] Completed {person_name} with {sample_count} valid samples.")

# Save the encodings and names to a pickle file
data = {"encodings": known_encodings, "names": known_names}

with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print("Encoding complete.")
print(f"Data saved to {ENCODINGS_FILE}")