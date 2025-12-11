import time
import matplotlib.pyplot as plt
import cv2
import numpy as np


def extract_edges_and_mask(image_path, red_thresh=50):
    # 1) Bild laden (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Bild konnte nicht geladen werden. Pfad prüfen!")

    # 2) Rotkanal extrahieren
    red = img[:, :, 2]

    # 3) Threshold -> Binärmaske (0/255)
    _, binary = cv2.threshold(red, red_thresh, 255, cv2.THRESH_BINARY)

    # 4) Kanten extrahieren (Canny)
    edges = cv2.Canny(binary, 50, 150)

    # 5) Kantenpixel koordinaten extrahieren
    ys, xs = np.where(edges > 0)

    # 6) linke / rechte Kante pro y
    left_edge = []
    right_edge = []

    for y in np.unique(ys):
        x_coords = xs[ys == y]
        if len(x_coords) >= 2:
            left_edge.append([int(np.min(x_coords)), int(y)])
            right_edge.append([int(np.max(x_coords)), int(y)])

    # 7) In numpy.ndarray wandeln
    to_red = np.array(left_edge, dtype=int)     # Shape (N,2)
    from_red = np.array(right_edge, dtype=int)  # Shape (N,2)

    return to_red, from_red, binary
def get_transitions(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {path}")

    # --- In HSV konvertieren ---
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_magenta = np.array([113, 43, 52])
    upper_magenta = np.array([165, 188, 203])
    mask_magenta = cv2.inRange(img_hsv, lower_magenta, upper_magenta)


    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    lower_red1 = np.array([0, 40, 40]);  upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40]); upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) | cv2.inRange(img_hsv, lower_red2, upper_red2)

 
    mask = cv2.bitwise_or(mask_magenta, mask_red)

    # --- Übergänge in X-Richtung suchen ---
    start = time.perf_counter()
    diff = np.diff(mask.astype(np.int8), axis=1)
    end = time.perf_counter()
    print(f"Transition detection duration: {end - start:.6f} s")

    # --- Pixelkoordinaten finden ---
    ys_enter, xs_enter = np.where(diff == +1)  # Übergang: Hintergrund -> Laser
    ys_leave, xs_leave = np.where(diff == -1)  # Übergang: Laser -> Hintergrund

    # +1, weil np.diff die Differenz zwischen Pixel (x-1, x) berechnet
    to_red = np.column_stack((xs_enter + 1, ys_enter))   # (x, y)
    from_red = np.column_stack((xs_leave + 1, ys_leave)) # (x, y)
    return to_red, from_red

import cv2, numpy as np, time


import cv2, numpy as np, time

def get_transitions_sensitive(path,
                              top_percent=0.05,   # Anteil der besten Score-Pixel, die behalten werden
                              border_drop=0.02,   # kleiner Randabzug
                              min_area=30,         # sehr kleine Fragmente verwerfen
                              gamma=1,         # <1 macht dunkles Rot sichtbarer
                              dilate_iter=0):     # verbindet Lücken minimal
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    # leichte Gamma-Korrektur
    imgf = np.power(np.clip(img.astype(np.float32)/255.0, 0, 1), gamma)
    img8 = (imgf*255).astype(np.uint8)

    # Merkmale
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    A = lab[:,:,1]  # Rot-Grün-Achse

    b, g, r = cv2.split(img8)
    r_excess = cv2.subtract(r, cv2.max(g, b))

    # Scores (alle 0..255)
    hue_dist  = cv2.min(cv2.absdiff(h, 0), cv2.absdiff(h, 180))
    hue_score = cv2.normalize(180 - hue_dist, None, 0, 255, cv2.NORM_MINMAX)
    rex_score = cv2.normalize(r_excess,       None, 0, 255, cv2.NORM_MINMAX)
    A_score   = cv2.normalize(A,              None, 0, 255, cv2.NORM_MINMAX)
    sv_score  = (s.astype(np.float32) * v.astype(np.float32) / 255.0).astype(np.uint8)

    # weiche Fusion (keine harten Gates)
    score = (0.45*rex_score.astype(np.float32) +
             0.25*hue_score.astype(np.float32) +
             0.20*A_score.astype(np.float32) +
             0.10*sv_score.astype(np.float32))

    # Kreis-ROI gegen Randringe
    H, W = score.shape
    if border_drop > 0:
        cx, cy = W/2.0, H/2.0
        R = 0.5 * min(H, W) * (1.0 - border_drop)
        yy, xx = np.ogrid[:H, :W]
        roi = ((xx - cx)**2 + (yy - cy)**2) <= R*R
        score *= roi

    # adaptiver Quantils-Threshold
    valid = score[score > 0]
    if valid.size == 0:
        return (np.empty((0,2), int), np.empty((0,2), int), np.zeros((H,W), np.uint8))
    t = np.percentile(valid, 100 - top_percent*100)
    mask = (score >= t).astype(np.uint8) * 255

    # minimal aufräumen + verbinden
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # kleine Komponenten entfernen
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # Übergänge
    diff = np.diff((clean > 0).astype(np.int8), axis=1)
    ys_enter, xs_enter = np.where(diff == +1)
    ys_leave, xs_leave = np.where(diff == -1)
    to_red   = np.column_stack((xs_enter + 1, ys_enter))
    from_red = np.column_stack((xs_leave + 1, ys_leave))
    return to_red, from_red, clean



    # --- Ergebnis zurückgeben ---
    return to_red, from_red
def pixel_to_cameraVector(u, v, K):
    pixelVector = np.array([u, v, 1])
    cameraVector = np.linalg.inv(K) @ (pixelVector )
    cameraVector = cameraVector / np.linalg.norm(cameraVector)
    return cameraVector
def get_intersection_Ray_Plane(cameraVector, plane_base, plane_normal):
    #Vereinfacht, da ursprung [0,0,0]
    lam = (plane_normal @ plane_base) / (plane_normal @ cameraVector)
    P = lam * cameraVector
    return P



#ab hier die neue variante:
import numpy as np
import cv2

def pixels_to_rays(uvs, K, dist=None):
    """
    uvs: (N,2) Pixelkoordinaten (x=u, y=v)
    Rückgabe: (N,3) normierte Richtungsvektoren v_c im Kamerasystem.
    """

    pts = np.asarray(uvs, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = pts.reshape(-1, 1, 2)

    if dist is None:
        # ohne Verzerrung: direkt mit inv(K)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        x = (uvs[:,0] - cx) / fx
        y = (uvs[:,1] - cy) / fy
        vc = np.stack([x, y, np.ones_like(x)], axis=1)
    else:
        # OpenCV gibt normierte Bildkoordinaten zurück (pp=(0,0), f=1)
        und = cv2.undistortPoints(pts, K, dist)  # (N,1,2)
        x = und[:,0,0]; y = und[:,0,1]
        vc = np.stack([x, y, np.ones_like(x)], axis=1)
    # normieren
    vc /= np.linalg.norm(vc, axis=1, keepdims=True)
    return vc
def intersect_rays_with_plane(v_c, plane_base_c, plane_normal_c, eps=1e-12):
    n = plane_normal_c.reshape(3)
    p0 = plane_base_c.reshape(3)
    num = np.dot(n, p0)                      # Skalar
    den = v_c @ n                            # (N,)
    den = np.where(np.abs(den) < eps, np.nan, den)  # Parallelfälle vermeiden
    lam = num / den                          # (N,)
    Pc = v_c * lam[:,None]                   # (N,3)
    return Pc
def cam_to_world_points(P_c, R_wc, t_wc):
    """
    P_c : (N,3) Punkte im Kamera-KS
    R_wc, t_wc : Welt-zu-Kamera Transform
    
    p_w = R_wc.T (p_c - t_wc)
    """
    P_c = np.asarray(P_c)
    return (R_wc.T @ (P_c.T - t_wc.reshape(3,1))).T


def check_frame(R):
    RtR = R.T @ R
    det = np.linalg.det(R)
    print("||R^T R - I||_F =", np.linalg.norm(RtR - np.eye(3)), "  det(R) =", det)

def rotz(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [ np.cos(angle), -np.sin(angle), 0 ],
        [ np.sin(angle),  np.cos(angle), 0 ],
        [ 0,              0,             1 ]
    ])
    return (R @ points.T).T  # gibt wieder (N,3) zurück
def rotx(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [1,             0,              0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    return (R @ points.T).T  # (N,3)
def roty(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [ 0,             1,             0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return (R @ points.T).T  # (N,3)

import cv2
import os
import numpy as np

def save_mask_overlay(clean, img_path):
    # Zielordner
    out_dir = "/home/janne/Desktop/Masterprojekt/Scan/last_scan_masks"
    os.makedirs(out_dir, exist_ok=True)

    # Originalbild laden
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {img_path}")

    # Overlay erzeugen (weiße Maske)
    overlay = img.copy()
    overlay[clean > 0] = (255, 255, 255)

    # Dateiname vorbereiten
    filename = os.path.basename(img_path)
    out_path = os.path.join(out_dir, filename)

    # Speichern
    cv2.imwrite(out_path, overlay)
    print(f"Overlay gespeichert unter: {out_path}")
