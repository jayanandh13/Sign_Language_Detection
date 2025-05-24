import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Constants
MODEL_PATH = './model.p'
LABELS_DICT = {0: 'A', 1: 'B', 2: 'L'}
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.5

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found - train the model first")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture")
    return cap

def initialize_hands_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

def process_frame(frame, hands, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    H, W = frame.shape[:2]
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

            # Extract and normalize landmarks
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

            # Create bounding box
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = LABELS_DICT[int(prediction[0])]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
    return frame

def main():
    try:
        print("Loading model...")
        model = load_model()
        
        print("Initializing camera...")
        cap = initialize_camera()
        
        print("Initializing hand tracking...")
        hands = initialize_hands_model()
        
        print("Starting real-time recognition. Press 'q' to quit...")
        
        prev_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Process frame
            frame = process_frame(frame, hands, model)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if 'hands' in locals():
            hands.close()
        print("Resources released")

if __name__ == "__main__":
    main()