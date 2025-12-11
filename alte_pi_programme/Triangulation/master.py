import triangulation_functions
import triangulation_transformation
import numpy as np
import os, shutil

scan_folder = r"/home/janne/Desktop/Masterprojekt/Scan/bracket"
data = np.load(r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_data_03.npz")


#hier leere ich den last masks ordner
out_dir = "/home/janne/Desktop/Masterprojekt/Scan/last_scan_masks"
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir, exist_ok=True)

mtx = data["mtx"]
dist = data["dist"]
centroid_L = data["centroid_L"]
normal_L = data["normal_L"]
centroid_R = data["centroid_R"]
normal_R = data["normal_R"]

print("mtx:\n", mtx)
print("normal_L:", normal_L)

#laden der drehteller daten
data_drehteller = np.load("/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_drehteller_mitkleineboard.npz")

center3d = data_drehteller["center3d"]
normal_drehteller = data_drehteller["normal_drehteller"]
punkte_drehteller = data_drehteller["pts"]
basis = data_drehteller["basis"]
u, v, n = basis[2], basis[1], normal_drehteller
u = u / np.linalg.norm(u)
v = v / np.linalg.norm(v)
n = -n / np.linalg.norm(n)

R_wc, t_wc = np.column_stack((u, v, n)), center3d
triangulation_functions.check_frame(R_wc)

Cw = triangulation_functions.cam_to_world_points(center3d.reshape(1,3), R_wc, t_wc)[0]
print("C ->", Cw)  # sollte ~ [0,0,0]

# Erwartet: kleine Zahl (~1e-15..1e-12) und det ≈ +1


import csv
import os

import os


# automatische Pfade
img_dir  = scan_folder
csv_path = os.path.join(scan_folder, "angles_capture_log.csv")

image_paths = []
angles = []

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row["filename"]
        angle    = float(row["angle_deg"])
        
        full_path = os.path.join(img_dir, filename)
        image_paths.append(full_path)
        angles.append(angle)
Pts_final = []
n=0
for path, angle in zip(image_paths, angles):
    print(path, angle)
    # 1) Pixel sammeln (to_red, from_red) -> (N,2)
    #to_red, from_red = triangulation_functions.get_transitions(path)
    to_red, from_red, mask_transitions = triangulation_functions.extract_edges_and_mask(path)
    triangulation_functions.save_mask_overlay(mask_transitions, path)
    if to_red.size == 0 or from_red.size == 0:
        print(f"[WARN] keine Übergänge erkannt in: {path}  → überspringe")
        continue
    #triangulation_functions.save_mask_overlay(mask_transition, path)
    uvs_L = to_red[:, :2].astype(float)
    uvs_R = from_red[:, :2].astype(float)

    



    # 2) Rays
    vL = triangulation_functions.pixels_to_rays(uvs_L, mtx, dist)   # (NL,3)
    vR = triangulation_functions.pixels_to_rays(uvs_R, mtx, dist)   # (NR,3)

    # 3) Schnittpunkte mit Laser-Ebenen (im Kamera-Frame)
    PL_c = triangulation_functions.intersect_rays_with_plane(vL, centroid_L, normal_L)  # (NL,3)
    PR_c = triangulation_functions.intersect_rays_with_plane(vR, centroid_R, normal_R)  # (NR,3)

    # 4) Alles zusammenführen
    Pts_c = np.vstack([PL_c, PR_c])  # (N,3) in Kamera-Koordinaten
    if n == 0:
        Pts_c = np.vstack([Pts_c, punkte_drehteller])
    n=n+1
    # 5) Einmaliger Sprung ins Welt-Frame
    # R_wc, t_wc z.B. aus deinem Drehteller-Fit: make_world_from_circle(center_c, normal_c, ...)
    Pts_w = triangulation_transformation.camtoworld(Pts_c, R_wc, t_wc)

    Pts_w_rotz = triangulation_functions.rotz(Pts_w, angle)

    Pts_final.append(Pts_w_rotz)
Pts_final = np.vstack(Pts_final)

# 1) optional ausdünnen (jeden k-ten Punkt behalten)
k = 10
Pts_final = Pts_final[::k]

# 2) nach Radius filtern (max 200 mm)
dist = np.linalg.norm(Pts_final, axis=1)
mask = (dist <= 75) & (Pts_final[:,2] > -20)
 
Pts_final_filtered = Pts_final[mask]


#punkte_drehteller_welt = triangulation_functions.cam_to_world_points(punkte_drehteller, R_wc, t_wc)
punkte_drehteller_welt = triangulation_transformation.camtoworld(punkte_drehteller, R_wc, t_wc)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# --- Drehtellerpunkte im Kamera- und Weltkoordinatensystem plotten ---

# Originalpunkte (Kamera)
points = np.asarray(Pts_final_filtered)

# Transformierte Punkte (Welt)
#points_world = triangulation_functions.cam_to_world_points(points_cam, R_wc, t_wc)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='red', label='gescannte Punkte Welt')

# Optional: Welt-Ursprung markieren
ax.scatter(0, 0, 0, c='green', s=80, label='Welt Ursprung (0,0,0)')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Drehtellerpunkte: Kamera vs Welt")
ax.legend()

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

import numpy as np

# Beispiel: Pts_final_filtered = (N×3) float-Array
np.savez("punkte_mein_scan.npz", P=Pts_final_filtered)

