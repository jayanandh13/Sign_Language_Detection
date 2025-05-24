import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found")

for dir_name in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_name)
    
    # Skip if not a directory (like .gitignore)
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory: {dir_path}")
        continue
    
    print(f"Processing class: {dir_name}")
    
    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)
        
        # Skip non-image files
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all landmarks first
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                # Normalize coordinates
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
            
            data.append(data_aux)
            labels.append(dir_name)
        else:
            print(f"No hand detected in {img_path}")

# Save the dataset
if not data:
    raise ValueError("No valid hand data was processed - check your input images")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created successfully with {len(data)} samples")