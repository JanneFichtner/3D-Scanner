#!/usr/bin/env python3
import os, time, cv2, numpy as np
from datetime import datetime
from picamera2 import Picamera2
import smbus

# --- Pfade & Setup ---
SAVE_DIR = "/home/janne/Desktop/Masterprojekt/Scan/bracket"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "angles_capture_log.csv")
I2C_BUS, DEVICE_AS5600 = 1, 0x36
bus = smbus.SMBus(I2C_BUS)

# --- Kamera/Steuerung ---
RESOLUTION = (640, 480)
WINDOW = "Scan Preview (B: manuelle Belichtung)"
ENTER_KEY, QUIT_KEY = 13, ord('q')
SETTLE_SEC = 0.30
DEBOUNCE_SEC = 0.25

# Startwerte & Limits (anpassbar)
EXP_US_MIN, EXP_US_MAX = 200, 1_000_000
GAIN_MIN,   GAIN_MAX   = 1.0, 16.0
exp_us, gain = 20_000, 1.5        # Startwerte
EXP_STEP_FACTOR = 1.25            # > / < skaliert Belichtungszeit
GAIN_STEP = 0.25                  # + / - ändert Gain

# --- AS5600 helpers ---
def read_raw_angle():
    d = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
    return ((d[0] << 8) | d[1]) & 0x0FFF

def read_magnitude():
    d = bus.read_i2c_block_data(DEVICE_AS5600, 0x1B, 2)
    return ((d[0] << 8) | d[1]) & 0x0FFF

def raw_to_deg(raw, raw_zero):
    rel = (raw + 4096 - raw_zero) & 0x0FFF
    return (rel * 360.0) / 4096.0

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def apply(cam, exp, g):
    cam.set_controls({"AeEnable": False, "ExposureTime": int(exp), "AnalogueGain": float(g)})

def draw_overlay(img, angle_deg, raw, mag, exp, g):
    txt1 = f"Angle: {angle_deg:7.2f} deg  (raw={raw:4d}, mag={mag:4d})"
    txt2 = f"Exp: {int(exp)} us   Gain: {g:.2f}   [Enter=Save | q=Quit | </>=Exp | +/-=Gain]"
    cv2.putText(img, txt1, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, txt2, (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def main():
    global exp_us, gain
    print("Starte Kamera & Sensor …")
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": RESOLUTION}))
    cam.start(); time.sleep(0.5)

    cam.set_controls({"AeEnable": False, "AwbEnable": True})
    apply(cam, exp_us, gain); time.sleep(SETTLE_SEC)

    raw_zero = read_raw_angle()
    print(f"AS5600 Nullpunkt gesetzt: raw_zero={raw_zero}")

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w") as f:
            f.write("timestamp,filename,angle_deg,raw_angle,magnitude,exposure_us,gain\n")

    print("Bedienung: [Enter]=Foto  [q]=Beenden  [</>]=Belichtungszeit  [+/-]=Gain")
    last_shot = 0.0

    try:
        while True:
            frame = cam.capture_array()
            raw = read_raw_angle(); mag = read_magnitude()
            angle_deg = raw_to_deg(raw, raw_zero)
            draw_overlay(frame, angle_deg, raw, mag, exp_us, gain)
            cv2.imshow(WINDOW, frame)
            k = cv2.waitKey(1) & 0xFF

            if k == QUIT_KEY: print("Beende …"); break

            # Gain: + / -
            if k in (ord('+'), ord('=')):  # '=' falls Shift nicht erfasst
                gain = clamp(gain + GAIN_STEP, GAIN_MIN, GAIN_MAX); apply(cam, exp_us, gain); time.sleep(SETTLE_SEC)
            elif k == ord('-'):
                gain = clamp(gain - GAIN_STEP, GAIN_MIN, GAIN_MAX); apply(cam, exp_us, gain); time.sleep(SETTLE_SEC)

            # Exposure: A / D
            elif k == ord('d'):   # erhöhen
                exp_us = clamp(int(exp_us * EXP_STEP_FACTOR), EXP_US_MIN, EXP_US_MAX)
                apply(cam, exp_us, gain)
                time.sleep(SETTLE_SEC)

            elif k == ord('a'):   # verringern
                exp_us = clamp(int(exp_us / EXP_STEP_FACTOR), EXP_US_MIN, EXP_US_MAX)
                apply(cam, exp_us, gain)
                time.sleep(SETTLE_SEC)

            # Enter → speichern
            elif k == ENTER_KEY:
                now = time.time()
                if now - last_shot < DEBOUNCE_SEC: continue
                last_shot = now
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"scan_angle_{angle_deg:07.2f}deg_exp{int(exp_us)}us_gain{gain:.2f}_{ts}.jpg"
                fpath = os.path.join(SAVE_DIR, fname)
                cam.capture_file(fpath)
                with open(CSV_PATH, "a") as f:
                    f.write(f"{ts},{fname},{angle_deg:.6f},{raw},{mag},{int(exp_us)},{gain:.4f}\n")
                print(f"Gespeichert: {fpath}")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    finally:
        try: cam.stop()
        except Exception: pass
        try: bus.close()
        except Exception: pass
        cv2.destroyAllWindows()
        print("Fertig.")

if __name__ == "__main__":
    main()
