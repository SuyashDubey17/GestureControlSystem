import cv2
import mediapipe as mp
import math
import time
import pyautogui
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)

# 🔥 BIGGER WINDOW (ONLY CHANGE)
cv2.namedWindow("FINAL STABLE SYSTEM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FINAL STABLE SYSTEM", 1400, 900)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

prev_vol = 50
prev_brightness = 0
pTime = 0

pinch_start_time = None
mute_triggered = False
active_mode = None


def draw_hand(frame, hand_landmarks, w, h, color):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    pts = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 3, color, -1)

    for c in connections:
        cv2.line(frame, pts[c[0]], pts[c[1]], color, 2)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result and result.hand_landmarks:

        for idx, hand_landmarks in enumerate(result.hand_landmarks):

            h, w, _ = frame.shape
            hand_label = result.handedness[idx][0].category_name

            color = (0,255,0) if hand_label=="Right" else (255,255,0)

            draw_hand(frame, hand_landmarks, w, h, color)

            x1 = int(hand_landmarks[8].x * w)
            y1 = int(hand_landmarks[8].y * h)
            x2 = int(hand_landmarks[4].x * w)
            y2 = int(hand_landmarks[4].y * h)

            length = math.hypot(x2-x1, y2-y1)

            if length < 10 or length > 250:
                continue

            if length < 45:
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),4)
            else:
                cv2.line(frame,(x1,y1),(x2,y2),color,3)

            if hand_label == "Right":

                active_mode = "VOLUME"

                value = int((length-30)/(170-30)*100)
                value = max(0,min(100,value))

                if length < 45:
                    if pinch_start_time is None:
                        pinch_start_time = time.time()
                    elif time.time() - pinch_start_time > 0.5 and not mute_triggered:
                        pyautogui.press("volumemute")
                        mute_triggered = True
                else:
                    pinch_start_time = None
                    mute_triggered = False

                diff = value - prev_vol

                if abs(diff) > 2:
                    if diff > 0:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")

                    prev_vol = value

            else:

                active_mode = "BRIGHTNESS"

                value = int((length-30)/(170-30)*100)
                value = max(0,min(100,value))

                sbc.set_brightness(value)
                prev_brightness = value

    if active_mode == "VOLUME":

        if prev_vol < 30:
            bar_color = (0,0,255)
        elif prev_vol < 70:
            bar_color = (0,255,255)
        else:
            bar_color = (0,255,0)

        bar = int(prev_vol*3)
        cv2.rectangle(frame,(50,150),(85,450),(100,100,100),2)
        cv2.rectangle(frame,(50,450-bar),(85,450),bar_color,-1)
        cv2.putText(frame,f'VOL {prev_vol}%',(40,480),
                    cv2.FONT_HERSHEY_COMPLEX,1,bar_color,2)

    elif active_mode == "BRIGHTNESS":

        bar = int(prev_brightness*3)
        cv2.rectangle(frame,(50,150),(85,450),(100,100,100),2)
        cv2.rectangle(frame,(50,450-bar),(85,450),(255,255,0),-1)
        cv2.putText(frame,f'BRT {prev_brightness}%',(40,480),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)

    if active_mode == "BRIGHTNESS":
        overlay = frame.copy()
        alpha = (100 - prev_brightness) / 100
        cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if mute_triggered:
        cv2.putText(frame,"MUTED",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cTime = time.time()
    fps = int(1/(cTime - pTime)) if (cTime - pTime)!=0 else 0
    pTime = cTime

    cv2.putText(frame,f'FPS: {fps}',(500,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
    cv2.putText(frame,f'FPS: {fps}',(500,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    cv2.putText(frame,"Gesture Control System",(150,30),
                cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),3)
    cv2.putText(frame,"Gesture Control System",(150,30),
                cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)

    cv2.putText(frame,"Pinch = Mute",(350,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
    cv2.putText(frame,"Pinch = Mute",(350,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,255,180),1)

    cv2.putText(frame,"Right: Volume | Left: Brightness",(250,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
    cv2.putText(frame,"Right: Volume | Left: Brightness",(250,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,255),1)

    cv2.imshow("FINAL STABLE SYSTEM", frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()