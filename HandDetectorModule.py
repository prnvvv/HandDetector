import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            print("Error: Could not open webcam.")
            exit()

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils


    def detect_hands(self):

        previousTime = 0
        currentTime = 0

        while True:

            ret, frame = self.cap.read()

            if not ret:

                print("Error: Could not read frame from webcam.")
                break

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(imgRGB)
            multipleHands = results.multi_hand_landmarks

            if multipleHands:

                for multipleHand in multipleHands:

                    for id, lm in enumerate(multipleHand.landmark):

                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(id, cx, cy)
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    self.mpDraw.draw_landmarks(frame, multipleHand, self.mpHands.HAND_CONNECTIONS)

            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 255), 3)

            cv2.imshow("Webcam Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    hand_detector = HandDetector()
    hand_detector.detect_hands()
