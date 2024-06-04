import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        multipleHands = results.multi_hand_landmarks

        if multipleHands:
            for multipleHand in multipleHands:
                if draw:
                    self.mpDraw.draw_landmarks(frame, multipleHand, self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(multipleHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return frame

def main():
    previousTime = 0

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = detector.findHands(frame)
        
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Webcam Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
