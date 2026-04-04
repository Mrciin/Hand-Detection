import cv2 as cv

def initialize_camera(index=0):
    """
    Initializes the camera capture.
    :param index: Camera index (default is 0 for built-in webcam).
    :return: cv2.VideoCapture object or None if failed.
    """
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {index}")
        return None
    return cap

def get_frame(cap):
    """
    Captures a single frame from the camera.
    :param cap: cv2.VideoCapture object.
    :return: The captured frame (numpy array) or None if failed.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?).")
        return None
    return frame

def release_camera(cap):
    """
    Releases the camera resource and closes any open OpenCV windows.
    :param cap: cv2.VideoCapture object.
    """
    if cap:
        cap.release()
    cv.destroyAllWindows()

def run_preview(camera_index=0):
    """
    A simple loop to test the camera functionality.
    """
    cap = initialize_camera(camera_index)
    if not cap:
        return

    print("Camera started. Press 'q' to exit.")
    try:
        while True:
            frame = get_frame(cap)
            if frame is None:
                break

            # You can perform processing here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('Camera Preview', gray)

            if cv.waitKey(1) == ord('q'):
                break
    finally:
        release_camera(cap)
