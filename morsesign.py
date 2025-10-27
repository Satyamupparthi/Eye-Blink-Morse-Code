import cv2
import mediapipe as mp
import time
import threading
from queue import Queue
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyttsx3

# ---------- TTS setup ----------
USE_WIN32 = False
try:
    import pythoncom
    import win32com.client
    USE_WIN32 = True
except Exception:
    USE_WIN32 = False

tts_queue = Queue()

def tts_worker_win32(q: Queue):
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        text = q.get()
        if text is None: break
        speaker.Speak(text)
    pythoncom.CoUninitialize()

def tts_worker_pyttsx3(q: Queue):
    engine = pyttsx3.init()
    while True:
        text = q.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
    engine.stop()

tts_thread = threading.Thread(target=tts_worker_win32 if USE_WIN32 else tts_worker_pyttsx3,
                              args=(tts_queue,), daemon=True)
tts_thread.start()

# ---------- Morse dictionary ----------
MORSE_CODE_DICT = {
    '.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F','--.':'G','....':'H','..':'I',
    '.---':'J','-.-':'K','.-..':'L','--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R',
    '...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y','--..':'Z'
}

# ---------- Mediapipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

blink_start = None
last_blink_time = time.time()
morse_sequence = ""
decoded_message = ""

# ---------- Stats ----------
blink_durations = []
decoded_letters = []
dot_count = 0
dash_count = 0
unrecognized_count = 0

# ---------- Video capture ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

def get_blink_ratio(landmarks, eye_points):
    top = landmarks[eye_points[1]].y
    bottom = landmarks[eye_points[5]].y
    left = landmarks[eye_points[0]].x
    right = landmarks[eye_points[3]].x
    vertical = abs(top - bottom)
    horizontal = abs(left - right)
    return vertical / horizontal if horizontal>0 else 1.0

print("Blink: short = '.', long = '-' | Press ESC to exit")

# ---------- Main Loop ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = [33,159,145,133,153,144]
            ratio = get_blink_ratio(landmarks, left_eye)

            if ratio < 0.20:  # eye closed
                if blink_start is None:
                    blink_start = time.time()
            else:
                if blink_start is not None:
                    dur = time.time() - blink_start
                    if dur >= 0.03:
                        blink_durations.append(dur)
                        if dur < 0.25:
                            morse_sequence += "."
                            dot_count += 1
                        else:
                            morse_sequence += "-"
                            dash_count += 1
                        print("[BLINK]", f"{dur:.2f}s ->", morse_sequence)
                    blink_start = None
                    last_blink_time = time.time()

        gap = time.time() - last_blink_time

        # End of letter
        if gap > 2.0 and morse_sequence:
            letter = MORSE_CODE_DICT.get(morse_sequence, '?')
            decoded_message += letter
            decoded_letters.append(letter)
            if letter=='?': unrecognized_count += 1
            print("[DECODE]", morse_sequence, "->", letter)
            tts_queue.put(letter)
            morse_sequence = ""

        # End of word
        if gap > 5.0 and decoded_message and not decoded_message.endswith(" "):
            decoded_message += " "
            print("[WORD]", decoded_message)

        cv2.putText(frame, f"Code: {morse_sequence}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        cv2.putText(frame, f"Message: {decoded_message}", (20,80), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        cv2.imshow("Eye Morse Code", frame)
        if cv2.waitKey(1)&0xFF==27: break

finally:
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    tts_thread.join(timeout=5)

# ---------- Plotting ----------
total_blinks = len(blink_durations)
total_letters = len(decoded_letters)
accuracy_pct = (total_letters - unrecognized_count) / total_blinks * 100 if total_blinks>0 else 0

sns.set_style("whitegrid")

# 1. Blink durations histogram
plt.figure(figsize=(6,4))
sns.histplot(blink_durations, bins=10, color="skyblue", edgecolor="black")
plt.title("Blink Duration Distribution")
plt.xlabel("Duration (s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Blink trend (blink index vs duration)
plt.figure(figsize=(6,4))
plt.plot(range(1,total_blinks+1), blink_durations, marker='o', color='purple')
plt.title("Blink Duration Trend")
plt.xlabel("Blink Index")
plt.ylabel("Duration (s)")
plt.tight_layout()
plt.show()

# 3. Dot vs Dash
plt.figure(figsize=(6,4))
sns.barplot(x=["Dots","Dashes"], y=[dot_count,dash_count], palette=["green","red"])
plt.title("Dot vs Dash Count")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 4. Letter recognition accuracy
plt.figure(figsize=(6,4))
plt.bar(["Recognized","Unrecognized"], [total_letters - unrecognized_count, unrecognized_count], color=["green","red"])
plt.title(f"Letter Recognition Accuracy ({accuracy_pct:.2f}%)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Letters over time
plt.figure(figsize=(8,4))
y_vals = [ord(c) for c in decoded_letters]
plt.plot(range(1, len(decoded_letters)+1), y_vals, marker='o', linestyle='-')
plt.yticks(y_vals, decoded_letters)
plt.xlabel("Letter Index")
plt.ylabel("Predicted Letter")
plt.title("Decoded Letters Sequence Over Time")
plt.tight_layout()
plt.show()

# 6. Cumulative Dots vs Dashes
cum_dots = np.cumsum([1 if dur < 0.25 else 0 for dur in blink_durations])
cum_dashes = np.cumsum([1 if dur >= 0.25 else 0 for dur in blink_durations])
plt.figure(figsize=(6,4))
plt.plot(range(1,total_blinks+1), cum_dots, label="Cumulative Dots", color="green")
plt.plot(range(1,total_blinks+1), cum_dashes, label="Cumulative Dashes", color="red")
plt.xlabel("Blink Index")
plt.ylabel("Count")
plt.title("Cumulative Dots vs Dashes Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Blink Duration vs Letter Type
letter_types = ['Dot' if dur < 0.25 else 'Dash' for dur in blink_durations]
plt.figure(figsize=(6,4))
sns.boxplot(x=letter_types, y=blink_durations, palette=["green","red"])
plt.title("Blink Duration by Letter Type")
plt.ylabel("Duration (s)")
plt.xlabel("Letter Type")
plt.tight_layout()
plt.show()

# 8. Word lengths over time
words = decoded_message.strip().split()
plt.figure(figsize=(6,4))
plt.plot(range(1,len(words)+1), [len(w) for w in words], marker='o', color='blue')
plt.xlabel("Word Index")
plt.ylabel("Letters in Word")
plt.title("Word Lengths Over Time")
plt.tight_layout()
plt.show()

# 9. Blink frequency
total_time = sum(blink_durations) + len(blink_durations)*0.1  # approx session time
blinks_per_min = len(blink_durations)/(total_time/60)
print(f"Average Blink Frequency: {blinks_per_min:.2f} blinks/min")
print(f"Total Blinks: {total_blinks} | Dots: {dot_count} | Dashes: {dash_count}")
print(f"Total Letters: {total_letters} | Unrecognized: {unrecognized_count}")
print(f"Accuracy: {accuracy_pct:.2f}%")
