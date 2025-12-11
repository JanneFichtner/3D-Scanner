import numpy as np
import cv2
import triangulation_functions
import triangulation_transformation
#laden der kalibierungsdaten
data = np.load(r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_data_01.npz")

mtx = data["mtx"]
dist = data["dist"]
centroid_L = data["centroid_L"]
normal_L = data["normal_L"]
centroid_R = data["centroid_R"]
normal_R = data["normal_R"]

print("mtx:\n", mtx)
print("normal_L:", normal_L)

#laden der drehteller daten
data_drehteller = np.load("/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_drehteller.npz")

base_drehteller = data_drehteller["center3d"]
normal_drehteller = data_drehteller["normal_drehteller"]
R_wc, t_wc = triangulation_transformation.make_world_from_circle(base_drehteller, normal_drehteller)

print("normale des drehteller bezüglich kamera: ", normal_drehteller)
path = r"/home/janne/Desktop/Masterprojekt/Kamera/Testbilder_belichtung/img_exp400000_gain2.00_idx16.jpg"
to_red, from_red = triangulation_functions.get_transitions(path)

# 1) Pixel sammeln (to_red, from_red) -> (N,2)
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

# 5) Einmaliger Sprung ins Welt-Frame
# R_wc, t_wc z.B. aus deinem Drehteller-Fit: make_world_from_circle(center_c, normal_c, ...)
Pts_w = triangulation_functions.cam_to_world_points(Pts_c, R_wc, t_wc)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

points = np.array(Pts_w)  # nur falls oben noch nicht ndarray

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



