import os
import cv2
import time

# Configuration
DATA_DIR = './data'
NUMBER_OF_CLASSES = 3  # A, B, L gestures
DATASET_SIZE = 100
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480  # Standard resolution
COUNTDOWN_SECONDS = 3  # Countdown before capturing

def setup_directories():
    """Create required directories if they don't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for class_id in range(NUMBER_OF_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(class_id))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def initialize_camera():
    """Initialize and configure the camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture device")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    return cap

def show_countdown(frame, seconds):
    """Display countdown on the frame"""
    for i in range(seconds, 0, -1):
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"Starting in {i}...", (150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame_copy)
        cv2.waitKey(1000)  # Wait 1 second between countdowns

def capture_images(cap, class_id):
    """Capture and save images for a specific class"""
    print(f'\nCollecting data for class {class_id}')
    print(f'Press "Q" when ready to start capturing {DATASET_SIZE} images...')

    # Wait for user to press Q
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == ord('q'):
            break

    # Countdown before capturing
    ret, frame = cap.read()
    if ret:
        show_countdown(frame, COUNTDOWN_SECONDS)

    # Capture images
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Display counter on frame
        frame_with_counter = frame.copy()
        cv2.putText(frame_with_counter, f"Capturing: {counter+1}/{DATASET_SIZE}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame_with_counter)
        
        # Save image
        img_path = os.path.join(DATA_DIR, str(class_id), f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
        
        # Small delay between captures
        if cv2.waitKey(25) == ord('q'):
            break

def main():
    try:
        setup_directories()
        cap = initialize_camera()
        
        for class_id in range(NUMBER_OF_CLASSES):
            capture_images(cap, class_id)
            
        print("\nData collection complete!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera resources released")

if __name__ == "__main__":
    main()