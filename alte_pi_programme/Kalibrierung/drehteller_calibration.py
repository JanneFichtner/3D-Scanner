import cv2
import numpy as np
import os
import glob
import calibration_functions
import matplotlib.pyplot as plt

# 1) Kalibrierung / Tvecs laden
result, kamera_parameters = calibration_functions.checkerboard_calibration(
    r"/home/janne/Desktop/Masterprojekt/encoder/Winkel Bilder"
)

# 2) Punkte sammeln (3D-Translationsvektoren)
pts = []
rotvecs = []
for item in result:
    t = np.asarray(item["translation"]).reshape(-1)  # (3,)
    pts.append(t)

    r = np.asarray(item["rotation"]).reshape(-1)  # (3,)
    rotvecs.append(r)
pts = np.array(pts)  # (N,3)
rotvecs = np.array(rotvecs)
mean_rotvec = rotvecs.mean(axis=0)
R, _ = cv2.Rodrigues(mean_rotvec)
board_normal_cam = R[:, 2]   # z-Achse des Boards → Normale

if len(pts) < 3:
    raise ValueError("Zu wenige valide Checkerboard-Posen für Kreis-Fit (<3).")
print("shape der rotvecs", rotvecs.shape)
print("erster rotvec", rotvecs[0])


# 3) Kreis in 3D fitten
center3d, normal, radius, rms, basis = calibration_functions.fit_circle_3d(pts)

print("Kreismittelpunkt (3D):", center3d)
print("Drehachsen-Richtung  :", normal)    # Ebenennormal (Einheitsvektor)
print("Radius               :", radius)
print("RMS-Fehler           :", rms)

# 4) Plot
calibration_functions.plot_circle_3d(pts, center3d, normal, radius, rotvecs, plane_basis=basis)

# 5) Speichern (klar benannte Keys)
save_path = r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_drehteller_mitkleineboard.npz"
np.savez(
    save_path,
    center3d=center3d,
    basis = basis,
    normal_drehteller=normal,   # hier habe ich ENDLICH die korrekte rotation hoffentlich
    radius=radius,
    rms=rms,
    pts=pts,
    rotvecs = rotvecs
)
print(f"Kalibrierungsdaten gespeichert unter:\n{save_path}")

def _set_equal_3d(ax, X):
    """Erzwingt gleiche Skalierung in 3D und cubic aspect für Punkte X (N,3)."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centers = (mins + maxs) / 2.0
    ranges = (maxs - mins)
    r = np.max(ranges) / 2.0
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)
    # echtes cubic aspect (Matplotlib >= 3.3)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

# Welt-Koordinatensystem definieren:
# - Ursprung im Kreismittelpunkt
# - Achsen als Orthonormalbasis [u, v, n] aus dem Fit
if basis is not None:
    p0, u, v = basis
    n = normal / np.linalg.norm(normal)
    print("basis is not None")
else:
    # Fallback: aus Normale u,v konstruieren
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)

R_world_cols = np.column_stack((u, v, n))      # 3x3, Spalten = Weltachsen im Kameraraum
origin_world = center3d.reshape(3,)            # Welt-Ursprung (Kamera-Raum-Koordinaten)

# Transform: Kamera → Welt: p_w = R_w^T (p_c - origin)
pts_world = (R_world_cols.T @ (pts - origin_world).T).T

fig = plt.figure(figsize=(12, 5))

# Links: Kamera-Raum, rot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pts[:,0], pts[:,1], pts[:,2], s=20, c='red', label='Kamera-Raum (rot)')
ax1.set_title('Messpunkte im Kamerakoordinatensystem')
ax1.set_xlabel('X_cam'); ax1.set_ylabel('Y_cam'); ax1.set_zlabel('Z_cam')
_set_equal_3d(ax1, pts)
ax1.legend()

# Rechts: Welt-Raum, blau
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pts_world[:,0], pts_world[:,1], pts_world[:,2], s=20, c='blue', label='Welt-Raum (blau)')
ax2.set_title('Messpunkte im Weltkoordinatensystem (u,v,n; Ursprung=Kreismittelpunkt)')
ax2.set_xlabel('X_w'); ax2.set_ylabel('Y_w'); ax2.set_zlabel('Z_w')
_set_equal_3d(ax2, pts_world)

# Optional: Weltachsen-Frame einzeichnen (kurze Pfeile)
L = radius * 0.5 if np.isfinite(radius) else 1.0
O = np.zeros(3)
ax2.quiver(O[0], O[1], O[2],  L, 0, 0, length=1.0, normalize=False, label='X_w')
ax2.quiver(O[0], O[1], O[2],  0, L, 0, length=1.0, normalize=False, label='Y_w')
ax2.quiver(O[0], O[1], O[2],  0, 0, L, length=1.0, normalize=False, label='Z_w')

ax2.legend()
plt.tight_layout()
plt.show()

