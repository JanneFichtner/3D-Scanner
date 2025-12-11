import cv2
import numpy as np
import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
def checkerboard_calibration(
        base,
        square_size = 4.444,
        CHECKERBOARD = (5,8)
):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints = [], []
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(g)

        # Ecken suchen
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        print(ret, "versuch:", path)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_size = gray.shape[::-1]  # (width, height)
            valid_paths.append(path)  # Nur die „guten“ Bilder merken

    # Kalibrieren
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, valid_size, None, None
    )
    result = []
    #Liste mit wichtigsten ergebnissen befüllen
    for path, rvec, tvec, corners in zip(valid_paths, rvecs, tvecs, imgpoints):
        result.append({
            "name": int(path[-6: -4]),
            "translation": tvec,
            "rotation": rvec,
            "points_l": None,
            "points_r": None,
            "shape": valid_size
        })
    kamera_parameters = {
        "mtx": mtx,
        "dist": dist
    }
    return result, kamera_parameters
def extract_bildnummer(path: str) -> int:
    # Nimmt die letzten zusammenhängenden Ziffern vor der Extension
    m = re.search(r'(\d+)(?=\.\w+$)', os.path.basename(path))
    return int(m.group(1)) if m else -1
def schnittpunkt_xAchse(x1, y1, x2, y2):
    # x-Koordinate des Schnittpunkts mit y=0 (x-Achse); np.inf bei vertikal/waagerecht
    if x2 == x1:
        return np.inf
    m = (y2 - y1) / (x2 - x1)
    if m == 0:
        return np.inf
    n = y1 - m * x1
    return (-n) / m
def get_laserline_points(base, mtx, dist, result,
                         use_undistort=True,
                         want_lines=2,
                         max_iters=50,
                         save_debug=True):
    
    patterns = [os.path.join(base, "**", "*.jpg"),
                os.path.join(base, "**", "*.jpeg"),
                os.path.join(base, "**", "*.png")]
    images = []
    for pat in patterns:
        images.extend(glob.glob(pat, recursive=True))
    print(f"{len(images)} Bilddateien gefunden.")

    debug_folder = r"/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/Makse_ueberpruefung"
    os.makedirs(debug_folder, exist_ok=True)


    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Überspringe (nicht lesbar): {path}")
            continue

        # Optional: Undistortion (nutzt mtx, dist)
        if use_undistort and mtx is not None and dist is not None:
            img = cv2.undistort(img, mtx, dist)

        # --- Farbfilter auf (meist) roten/magentafarbenen Laser ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Option A: deine Magenta-Range (ggf. anpassen)
        lower_magenta = np.array([113, 43, 52])
        upper_magenta = np.array([165, 188, 203])
        mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)

        # Option B: klassischer Rot-Laser: zwei Hue-Bereiche (0-10) und (170-180)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Nimm vereint (oder nur eine der beiden, je nach Laserfarbe)
        mask = cv2.bitwise_or(mask_magenta, mask_red)

        # Kleine Artefakte entfernen / Linien schließen
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- Debug: Maske speichern ---
        if save_debug:
            bildnummer = extract_bildnummer(path)
            mask_path = os.path.join(debug_folder, f"mask_{bildnummer:02d}.png")
            cv2.imwrite(mask_path, mask)

        edges = cv2.Canny(mask, 50, 150, apertureSize=7)

        # Adaptive Hough-Parameter: starte moderat, passe an
        rho = 1
        theta = np.pi / 360
        thresh = 100
        minLine = 120         # je nach Bildskalierung anpassen
        maxGap  = 500

        linesP = cv2.HoughLinesP(mask, rho, theta, thresh,
                                 minLineLength=minLine, maxLineGap=maxGap)

        iters = 0
        # Falls keine oder „zu viele“ Linien: justiere Threshold ein paar Male.
        # WICHTIG: Keine Endlosschleife, und niemals len(None).
        while iters < max_iters and (linesP is None or len(linesP) < want_lines):
            iters += 1
            thresh = max(1, thresh - 2)   # sensibler machen
            linesP = cv2.HoughLinesP(mask, rho, theta, thresh,
                                     minLineLength=minLine, maxLineGap=maxGap)

        # Wenn sehr viele Linien, erhöhen wir Threshold ein paar Male, um Ausreißer zu reduzieren
        while iters < max_iters and linesP is not None and len(linesP) > 10:
            iters += 1
            thresh = min(1000, thresh + 5)   # strenger machen
            linesP = cv2.HoughLinesP(mask, rho, theta, thresh,
                                     minLineLength=minLine, maxLineGap=maxGap)

        bildnummer = extract_bildnummer(path)

        if linesP is None or len(linesP) == 0:
            print(f"Die Hough Trafo konnte keine Linien ermitteln bei Bild: {bildnummer} (iter={iters}, thr={thresh})")
            # nichts zuzuordnen; weiter
            continue

        # --- aus allen gefundenen Segmenten die besten 2 wählen ---
        # Kriterium: längste Linien (Pixel-Länge)
        lines = linesP[:, 0]  # shape (N,4)
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
        idx = np.argsort(-lengths)[:max(want_lines, 2)]
        top2 = lines[idx]

        
        if len(top2) < 2:
            print(f"Nur {len(top2)} Linie(n) brauchbar bei Bild: {bildnummer}")
            continue

        (x1, y1, x2, y2), (u1, v1, u2, v2) = top2

                # --- Debug: Linien ins Bild plotten und speichern ---
        if save_debug:
            overlay = img.copy()
            # Linie 1 = grün
            cv2.line(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
            # Linie 2 = rot
            cv2.line(overlay, (u1, v1), (u2, v2), (0,0,255), 2)

            overlay_path = os.path.join(debug_folder, f"overlay_{bildnummer:02d}.png")
            cv2.imwrite(overlay_path, overlay)


        sx1 = schnittpunkt_xAchse(x1, y1, x2, y2)
        sx2 = schnittpunkt_xAchse(u1, v1, u2, v2)

        # ins result-Array mappen
        for entry in result:
            if entry["name"] == bildnummer:
                # Fallback: falls beide sx inf sind (beide vert/horiz), nimm die mit kleinerem x-Mittel als "links"
                if np.isinf(sx1) and np.isinf(sx2):
                    midx1 = 0.5 * (x1 + x2)
                    midx2 = 0.5 * (u1 + u2)
                    if midx1 <= midx2:
                        entry["points_l"] = [(x1, y1), (x2, y2)]
                        entry["points_r"] = [(u1, v1), (u2, v2)]
                    else:
                        entry["points_l"] = [(u1, v1), (u2, v2)]
                        entry["points_r"] = [(x1, y1), (x2, y2)]
                else:
                    # Deine Logik per x-Achsen-Schnitt
                    if sx1 > sx2:
                        entry["points_l"] = [(u1, v1), (u2, v2)]
                        entry["points_r"] = [(x1, y1), (x2, y2)]
                    else:
                        entry["points_l"] = [(x1, y1), (x2, y2)]
                        entry["points_r"] = [(u1, v1), (u2, v2)]
                break

        print(f"{bildnummer} hat {iters} Iterationen gebraucht (th={thresh}, Linien={len(linesP)})")

    return result
def pixel_to_cameraVector(u, v, K):
    pixelVector = np.array([u, v, 1])
    cameraVector = np.linalg.inv(K) @ (pixelVector )
    cameraVector = cameraVector / np.linalg.norm(cameraVector)
    return cameraVector
def get_normal_Checkerboard(rvec):
    #Normalenvektor des checkerboards ist aus R transformierte Z-Achse
    R, _ = cv2.Rodrigues(rvec)
    z= np.array([0, 0, 1])
    normal = R @ z
    return normal / np.linalg.norm(normal)
def get_intersection(cameraVector, tvec, normal):
    #Vereinfacht, da ursprung [0,0,0]
    lam = (normal @ tvec) / (normal @ cameraVector)
    P = lam * cameraVector
    return P
def fit_plane_svd(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1, :]
    return centroid, normal / np.linalg.norm(normal)
def plane_local_basis(n: np.ndarray):
    """Erzeuge zwei orthogonale Einheitsvektoren (u,v) in der Ebene ⟂ n."""
    n = n / np.linalg.norm(n)
    # wähle Hilfsvektor, der nicht parallel zu n ist
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    return u, v
def fit_plane_svd(points3d: np.ndarray):
    """points3d: (N,3). Return: plane_point p0 (centroid), unit normal n."""
    P = np.asarray(points3d, dtype=float).reshape(-1, 3)
    p0 = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - p0, full_matrices=False)
    n = vh[-1]                         # kleinste Singulärrichtung
    n = n / np.linalg.norm(n)
    return p0, n
def fit_circle_2d(xy: np.ndarray):
    """
    Algebraischer Least-Squares-Kreisfit (Kåsa/Taubin-Variante).
    xy: (N,2). Return center (cx,cy), radius r.
    Gleichung: x^2 + y^2 = 2a x + 2b y + c  ->  solve [2x,2y,1] [a,b,c]^T = x^2+y^2
    """
    X = np.asarray(xy, dtype=float)
    A = np.c_[2*X[:,0], 2*X[:,1], np.ones(len(X))]
    b = (X[:,0]**2 + X[:,1]**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol
    cx, cy = a, b_
    r = np.sqrt(cx*cx + cy*cy + c)
    return np.array([cx, cy]), float(r)
def fit_circle_3d(points3d: np.ndarray, robust=True, zscore=2.5):
    """
    Fit Kreis in 3D:
      - Ebene per SVD
      - Projektion in (u,v)
      - 2D-Kreisfit
      - Rückprojektion
    Optional: 1x Outlier-Reject via Z-Score auf Kreisabstand.
    Returns: center3d, normal(unit), radius, rms, (p0,u,v)
    """
    P = np.asarray(points3d, dtype=float).reshape(-1, 3)

    # 1) Ebene
    p0, n = fit_plane_svd(P)
    u, v = plane_local_basis(n)

    # 2) Projektion in 2D
    Q = P - p0
    xy = np.c_[Q @ u, Q @ v]

    # 3) erster Kreisfit
    (cx, cy), r = fit_circle_2d(xy)

    # 4) optional: simple robust step (einmal ausreißer verwerfen)
    if robust:
        d = np.abs(np.sqrt((xy[:,0]-cx)**2 + (xy[:,1]-cy)**2) - r)
        m, s = np.median(d), (1.4826*np.median(np.abs(d - np.median(d))) + 1e-12)
        keep = d <= (m + zscore*s)
        if keep.sum() >= 3 and keep.sum() < len(d):
            (cx, cy), r = fit_circle_2d(xy[keep])

    # 5) zurück in 3D
    center3d = p0 + cx*u + cy*v

    # 6) Fehler
    rad = np.sqrt((xy[:,0]-cx)**2 + (xy[:,1]-cy)**2)
    rms = float(np.sqrt(np.mean((rad - r)**2)))

    return center3d, n/np.linalg.norm(n), r, rms, (p0, u, v)
def plot_points(points):


    points = np.array(points)  # nur falls oben noch nicht ndarray

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)  # s = Punktgröße

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Triangulierte Punkte")

    # --- helper: gleiche Skalen auf allen Achsen ---
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    set_axes_equal(ax)

    plt.show()
def plot_circle_3d(points, center3d, normal, radius, rotvecs, plot_normals=True, plane_basis=None, axis_scale=1.2):
    """
    points      : (N,3) Originalpunkte
    center3d    : Mittelpunkt des Kreises in 3D
    normal      : Ebenennormalenvektor (Einheitslänge)
    radius      : Kreisradius
    plane_basis : (p0,u,v) aus fit_circle_3d() (optional, aber empfohlen)
    axis_scale  : Darstellungslimit im Plot
    """

    # lokale Basis der Ebene wiederverwenden (falls vorhanden)
    if plane_basis is not None:
        p0, u, v = plane_basis
    else:
        # fallback-Basis erzeugen
        n = normal / np.linalg.norm(normal)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u /= np.linalg.norm(u)
        v = np.cross(n, u)

    # Kreis generieren (in 3D)
    t = np.linspace(0, 2*np.pi, 200)
    circle3d = center3d + radius*np.outer(np.cos(t), u) + radius*np.outer(np.sin(t), v)

    # Achse als Linie darstellen
    axis_len = radius * axis_scale
    A = center3d - axis_len*normal
    B = center3d + axis_len*normal

    # 3D-Plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Punkte
    P = np.asarray(points)
    ax.scatter(P[:,0], P[:,1], P[:,2], s=25, color='blue', label="Messpunkte")

    # Kreis
    ax.plot(circle3d[:,0], circle3d[:,1], circle3d[:,2], color='red', label="Gefitteter Kreis")

    # Mittelpunkt
    ax.scatter(center3d[0], center3d[1], center3d[2], color='green', s=80, label="Kreismittelpunkt")

    # Rotationsachse
    ax.plot([A[0],B[0]],[A[1],B[1]],[A[2],B[2]], color='black', linewidth=2, label="Rotationsachse")

    if plot_normals:
        for point, rvec in zip(points, rotvecs):
            s = point
            R, _ = cv2.Rodrigues(rvec)
            normal = R[:, 2]   # z-Achse des Boards → Normale
            e = s + normal * 10
            ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]], color='blue', linewidth=2)



    # Achsenbeschriftung
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Kreisfit auf Drehtellerpunkten")

    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # <-- erzwingt gleiches Seitenverhältnis im 3D
        # *** GLEICHE ACHSSKALIERUNG ***
    all_pts = np.vstack([P, circle3d, center3d.reshape(1,3), A.reshape(1,3), B.reshape(1,3)])
    x_limits = [all_pts[:,0].min(), all_pts[:,0].max()]
    y_limits = [all_pts[:,1].min(), all_pts[:,1].max()]
    z_limits = [all_pts[:,2].min(), all_pts[:,2].max()]
    max_range = max(x_limits[1]-x_limits[0],
                    y_limits[1]-y_limits[0],
                    z_limits[1]-z_limits[0]) / 2

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # *** Ende Equal-Scaling-Block ***

    plt.tight_layout()
    plt.show()