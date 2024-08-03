# Face Detection App

This is a simple web application for face detection using OpenCV and Streamlit. The app can detect faces from a live camera feed or from uploaded images.

## Features

- Detect faces from live camera feed
- Detect faces from uploaded images
- Display the detected faces with bounding boxes

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/whothefisyash/Open-CV--Face-Detection.git
    cd face-detection-app
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:
    
        ```sh
        .\venv\Scripts\activate
        ```

    - On macOS/Linux:
    
        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

2. **Open your browser and go to:**

    ```
    http://localhost:8501
    ```

3. **Use the App:**

    - Click on the "Open Camera" button to start face detection from your webcam.
    - Upload an image to detect faces in the image.

## Troubleshooting

### Common Issues

1. **Error: Could not open camera:**

    - Ensure that no other application is using the camera.
    - Check if the camera is properly connected and recognized by your operating system.
    - Try using different camera indices if you have multiple cameras.

2. **Error: Could not read frame:**

    - This typically means the camera is not returning frames. Ensure the camera is functioning correctly.

3. **Permissions Issues:**

    - Ensure the app has permission to access the camera.
    - On Windows, go to Settings > Privacy > Camera and ensure the app has permission.
    - On macOS, go to System Preferences > Security & Privacy > Camera and ensure the app has permission.

### Testing Camera

You can run the following simple script to test if your camera is working outside of Streamlit:

```python
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
