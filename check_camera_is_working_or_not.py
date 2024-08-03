import cv2

def test_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index {index}")
            break
        else:
            cap.release()
    else:
        print("Error: Could not open any camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Camera Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

test_camera()
