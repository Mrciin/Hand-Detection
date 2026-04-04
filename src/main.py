import cv2 as cv
from camera_parsing import initialize_camera, get_frame, release_camera
from hand_tracker import HandTracker

def main():
    # 1. Initialize c1amera and tracker
    cap = initialize_camera(0)
    tracker = HandTracker(detection_con=0.7, track_con=0.7)
    
    if not cap:
        print("Failed to start the application.")
        return

    print("Application started. Press 'q' to quit.")

    try:
        while True:
            # 2. Capture a frame
            frame = get_frame(cap)
            if frame is None:
                break

            # 3. Detect hands and draw landmarks
            frame = tracker.find_hands(frame)
            
            # 4. (Optional) Get specific landmark data for logic
            landmarks = tracker.get_landmarks(frame, draw=False)
            if landmarks:
                # Example: Index finger tip is landmark 8
                finger_id, x, y = landmarks[8]
                cv.putText(frame, f"Index Tip: {x}, {y}", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 5. Display the final result
            cv.imshow("Hand Detection System", frame)

            # 6. Exit on 'q' key
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        # 7. Clean up resources
        release_camera(cap)
        print("Application closed safely.")

if __name__ == "__main__":
    main()
