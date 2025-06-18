import cv2
import mediapipe as mp
import numpy as np
import os

# Debug info
print('Current working directory:', os.getcwd())
print('Files in current directory:', os.listdir())

# Load template image
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
if template is None:
    print('Error: hand_template.png not found in current directory.')
    exit(1)
template_h, template_w = template.shape[:2]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Cannot access webcam.')
    exit(1)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Overlay function (with boundary check)
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img.shape[:2]
    oh, ow = img_overlay.shape[:2]

    # Avoid overflow
    if x < 0 or y < 0 or x + ow > w or y + oh > h:
        return

    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        img[y:y+oh, x:x+ow, c] = (
            alpha_overlay * img_overlay[:, :, c] +
            alpha_background * img[y:y+oh, x:x+ow, c]
        )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Error: Cannot read frame from webcam.')
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    overlay_image_alpha(frame, template, (center_x, center_y))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):
                print('Hand detected inside template area. Playing video...')
                video = cv2.VideoCapture('vid.mp4')
                if not video.isOpened():
                    print('Error: vid.mp4 not found.')
                    continue
                while video.isOpened():
                    ret_vid, frame_vid = video.read()
                    if not ret_vid:
                        break
                    cv2.imshow('Hand Scanner', frame_vid)
                    if cv2.waitKey(30) & 0xFF == 27:
                        break
                video.release()

    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
