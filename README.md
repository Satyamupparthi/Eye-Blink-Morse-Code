# ????️ Eye Blink Morse Code Communication System

### Hands-Free Communication with Real-time Computer Vision
A Python system that identifies **eye blinks** through **Mediapipe FaceMesh** and converts them into **Morse code**.
The Morse code is then deciphered and translated into **speech** by text-to-speech synthesis — providing hands-free, voice-enabled communication.

---

## ???? Overview
This project touches on the convergence of **AI, computer vision, and accessibility**.
By correlating the fine timing of eye blinks with Morse code (short blinks = dots, long blinks = dashes), the system offers a new means for users to "type" and "speak" using nothing but their eyes. 

It's not an app — it's a **research-grade prototype** showing how machine perception and adaptive signal processing can enable new human–computer interaction techniques.

---

## ???? Features
- ????️ **Real-time blink detection** through Mediapipe FaceMesh
- ???? **Decoding of Morse code** from blink lengths
- ????️ **Speech synthesis** via pyttsx3 (cross-platform) or Windows SAPI
- ???? **Data analysis and visualizations** (blink patterns, recognition rate, and cumulative signals)
- ⚙️ **Threaded text-to-speech** system with queue synchronization
- ???? **Session logging and performance statistics** for offline analysis

----

## ???? Tech Stack
| Category | Technologies |
|-----------|---------------|
| Language | Python |
| Computer Vision | OpenCV, Mediapipe |
| Data Processing | NumPy |
| Visualization | Matplotlib, Seaborn |
| Speech Synthesis | pyttsx3, win32com (optional) |
| Utilities | threading, Queue, time, sys |

---

