import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hand model and PyAutoGUI
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV window
cap = cv2.VideoCapture(1)

screen_width, screen_height = pyautogui.size()
smoothening = 5  # Adjust smoothening for stable movement

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Function to smooth the mouse movement using low-pass filtering
def smooth_movement(x, y, prev_x, prev_y, smoothening):
    curr_x = prev_x + (x - prev_x) / smoothening
    curr_y = prev_y + (y - prev_y) / smoothening
    return curr_x, curr_y

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame to avoid mirrored movement
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as Mediapipe requires it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Extract landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the landmarks for index finger tip (Landmark 8) and thumb tip (Landmark 4)
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert normalized coordinates to screen coordinates
            x_index = int(index_finger_tip.x * screen_width)
            y_index = int(index_finger_tip.y * screen_height)

            x_thumb = int(thumb_tip.x * screen_width)
            y_thumb = int(thumb_tip.y * screen_height)

            # Calculate the distance between the thumb and the index finger
            distance = np.hypot(x_thumb - x_index, y_thumb - y_index)

            # If the distance is large, move the cursor
            if distance > 30:  # Cursor moves only when fingers are apart
                curr_x, curr_y = smooth_movement(x_index, y_index, prev_x, prev_y, smoothening)
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # If the distance is small, simulate a mouse click (fingers close together)
            if distance < 20:  # Clicking threshold
                pyautogui.click()

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
