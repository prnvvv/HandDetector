# Import necessary libraries
import cv2
import mediapipe as mp
import time

# Class to detect and track hands using MediaPipe
class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    # Method to detect hands in an image
    def detectHands(self, img, draw=True):
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Draw hand landmarks on the image if any hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # Method to find and return the positions of hand landmarks
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lmList

# Main function to capture video and process each frame
def main():
    previousTime = currentTime = 0

    # Start video capture from webcam
    capture = cv2.VideoCapture(0)

    # Create an instance of HandDetector
    detector = HandDetector()

    while True:
        success, vidObject = capture.read()
        if not success:
            break

        # Detect hands in the frame
        vidObject = detector.detectHands(vidObject)

        # Find positions of hand landmarks
        lmList = detector.findPosition(vidObject)
        if len(lmList) != 0:
            print(lmList[4])  # Print the position of the thumb tip

        # Calculate frames per second (FPS)
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Display FPS on the video
        cv2.putText(vidObject, f"FPS : {int(fps)}", (40, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
        cv2.imshow("Video", vidObject)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()

# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
