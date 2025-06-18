from flask import Flask, render_template, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import os
import time

app = Flask(__name__)

# Global variables for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
template_path = os.path.join(os.path.dirname(__file__), 'hand_template.png')
template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2] if template is not None else (0, 0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        last_time = 0
        fps = 10  # Limit FPS to 10 for smoothness and less lag
        while True:
            now = time.time()
            if now - last_time < 1.0 / fps:
                time.sleep(max(0, 1.0 / fps - (now - last_time)))
            last_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if template is not None:
                center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2
                # Overlay template
                if template.shape[2] == 4:
                    alpha_overlay = template[:, :, 3] / 255.0
                    alpha_background = 1.0 - alpha_overlay
                    for c in range(3):
                        frame[center_y:center_y+template_h, center_x:center_x+template_w, c] = (
                            alpha_overlay * template[:, :, c] +
                            alpha_background * frame[center_y:center_y+template_h, center_x:center_x+template_w, c]
                        )
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        cap.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play_video')
def play_video():
    def gen():
        video_path = os.path.join(os.path.dirname(__file__), 'vid.mp4')
        video = cv2.VideoCapture(video_path)
        frame_delay = 0.05  # 20 FPS (slower playback)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break  # Stop when video ends
            frame = cv2.resize(frame, (640, 480))
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(frame_delay)
        video.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan_hand')
def scan_hand():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detected = False
    ret, frame = cap.read()
    if ret and template is not None:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                if (center_x < x_min < center_x + template_w and
                    center_y < y_min < center_y + template_h and
                    center_x < x_max < center_x + template_w and
                    center_y < y_max < center_y + template_h):
                    detected = True
    cap.release()
    if detected:
        return jsonify({'result': 'play'}), 200
    else:
        return jsonify({'result': '❌ No hand detected in template area.'}), 200

if __name__ == '__main__':
    app.run(debug=True)
