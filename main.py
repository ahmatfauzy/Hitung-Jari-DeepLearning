import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
finger_images = [cv2.imread(f"./jari/{i}.jpg") for i in range(1, 6)]

for i in range(5):
    finger_images[i] = cv2.resize(finger_images[i], (100, 100))

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_landmarks = hand_landmarks.landmark
            
            finger_tips = [4, 8, 12, 16, 20]
            knuckles = [3, 7, 11, 15, 19]

            fingers_up = sum(1 for tip, knuckle in zip(finger_tips, knuckles) if finger_landmarks[tip].y < finger_landmarks[knuckle].y)
            fingers_down = 5 - fingers_up

            print(f"Fingers Up: {fingers_up}, Fingers Down: {fingers_down}")
            if fingers_up > 0 and fingers_up <= 5:
                cv2.imshow("Finger Image", finger_images[fingers_up - 1])

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
