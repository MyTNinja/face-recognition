import numpy as np
from keras.preprocessing import image
import cv2
from model import classifier, ResultMap

'''########### Making single predictions ###########'''

# Load the pre-trained Haarcascade classifier for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture from webcam (change 0 to the appropriate camera index if needed)
cap = cv2.VideoCapture(0)

# Initialize variables

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face region from the frame
        cropped_face = frame[y:y+h, x:x+w]
        cv2.imwrite('cropped_face.jpg', cropped_face)
        # ImagePath = 'data/Final Testing Images/image_0415_Face_1.jpg'
        ImagePath = 'cropped_face.jpg'
        test_image = image.load_img(ImagePath, target_size=(64, 64))
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        result = classifier.predict(test_image, verbose=0)
        print(result)

        if np.max(result)>0.8:
            cv2.putText(frame, ResultMap[np.argmax(result)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Match', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video Stream', frame)

    # Wait for 'q' key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
