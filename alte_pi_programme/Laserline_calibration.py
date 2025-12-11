import numpy as np
import os
import glob
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Innere Ecken (Anzahl Schnittpunkte, nicht Felder!)
CHECKERBOARD = (3, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints, imgpoints = [], []

square_size = 14.64  # mm
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
#DateFrame mit Spalten definieren

df = pd.DataFrame(columns=["path", "img", "translation", "rotation", "points"])

# --- Pfade korrekt globs’en (hier rekursiv .jpg/.png) ---
base = r"/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/checkerboard_light"
patterns = [os.path.join(base, "**", "*.jpg"),
            os.path.join(base, "**", "*.jpeg"),
            os.path.join(base, "**", "*.png")]
images = []
for pat in patterns:
    images.extend(glob.glob(pat, recursive=True))
print(f"{len(images)} Bilddateien gefunden.")

valid_size = None
valid_paths = []  # Liste für die Bilder, die tatsächlich Ecken liefern

for path in images:
    img = cv2.imread(path)  # BGR laden
    if img is None:
        print(f"Überspringe (nicht lesbar): {path}")
        continue

    # Grünkanal nehmen, um roten Laser zu minimieren
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(g)

    # Ecken suchen
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    print(ret, "versuch:", path)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        valid_size = gray.shape[::-1]  # (width, height)
        valid_paths.append(path)       # Nur die „guten“ Bilder merken

# Kalibrieren
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, valid_size, None, None
)

print("Camera matrix:\n", mtx)
print(f"Es konnten {len(imgpoints)} von {len(images)} Bildern genutzt werden.")

# Jetzt DataFrame befüllen
rows = []
for path, rvec, tvec, corners in zip(valid_paths, rvecs, tvecs, imgpoints):
    rows.append({
        "path": path,
        "translation": tvec,
        "rotation": rvec,
        "points": None,
        "shape": valid_size
    })

df = pd.DataFrame(rows)



print("Camera matrix:\n", mtx)
print("Es konnten ", len(imgpoints), " von ", len(images), " Bildern genutzt werden.")
print("Translationsvektor: ", tvecs[-1],"\n", "Rotationsvektor: ", rvecs[-1])

import os
base = r"/home/janne/Desktop/Masterprojekt"
# Zielpfad im Projektordner (z. B. neben deinem Notebook)
save_path = os.path.join(base, "calibration_results_thinlaser.pkl")

# DataFrame speichern
df.to_pickle(save_path)

print(f"✅ DataFrame erfolgreich gespeichert unter:\n{save_path}")
