# Import necessary libraries
import cv2
import mediapipe as mp
import time

# Open a connection to the webcam (usually the first webcam is at index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands solution
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Variables to calculate frames per second (FPS)
previousTime = 0
currentTime = 0

# Infinite loop to process each frame from the webcam
while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(imgRGB)
    multipleHands = results.multi_hand_landmarks

    # Draw landmarks if hands are detected
    if multipleHands:
        for multipleHand in multipleHands:
            for id, lm in enumerate(multipleHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)  # Print landmark ID and its coordinates
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Draw circle on each landmark
            mpDraw.draw_landmarks(frame, multipleHand, mpHands.HAND_CONNECTIONS)

    # Calculate frames per second (FPS)
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Display the FPS on the frame
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    # Display the processed frame
    cv2.imshow("Webcam Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
