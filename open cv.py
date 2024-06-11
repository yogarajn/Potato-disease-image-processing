import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define constants
IMAGE_SIZE = (224, 224)
DATASET_PATH = r'C:\Users\ASUS VivoBook 15\Desktop\New folder'

# Load the trained model
model = load_model(r'C:\Users\ASUS VivoBook 15\Desktop\MyModelFolder\my_model.keras')

# Load dataset to get class names
dataset = image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

# Get class names
class_names = dataset.class_names

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Process the largest contour if it has a significant area
    if contours and cv2.contourArea(contours[0]) > 500:  # Adjust the threshold as needed
        contour = contours[0]

        # Get the bounding box for the largest contour and expand it slightly
        x, y, w, h = cv2.boundingRect(contour)
        padding = 10  # Add padding to ensure the whole leaf is covered
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, frame.shape[1] - x)
        h = min(h + 2 * padding, frame.shape[0] - y)

        # Extract the region of interest (ROI) and preprocess it for prediction
        roi = frame[y:y+h, x:x+w]
        processed_roi = cv2.resize(roi, (224, 224))
        processed_roi = processed_roi / 255.0
        processed_roi = np.expand_dims(processed_roi, axis=0)

        # Make prediction
        prediction = model.predict(processed_roi)
        predicted_class = np.argmax(prediction)

        # Draw a rectangle around the detected leaf and display the class name
        if predicted_class != 0:  # Assuming class 0 is background (no object)
            class_name = class_names[predicted_class]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)  # Change the color as needed
            cv2.putText(frame, "Class: {}".format(class_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the rectangles
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
