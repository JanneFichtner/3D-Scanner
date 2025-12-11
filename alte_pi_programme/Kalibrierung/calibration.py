"""
Kalibrierung für den Rasberry PI
Was brauche ich, um die bilder zu triangulieren?
Kameraparameter: mtx, dist, evtl newmtx
    -hierfür die 10 bilder aus checkerboard light laden und die kameraparameter
    -tvecs und rvecs für lasererkennung speichern
Die beiden Ebenen des Lasers, können durch die base und normal definiert werden.

am ende muss ich mtx, dst und centroid und normal in einer datei speichern
"""
import numpy as np

import calibration_functions
checkerboard_light_ordner = r"/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/checkerboard_light"
result, kamera_parameters = calibration_functions.checkerboard_calibration(checkerboard_light_ordner)

laser_dark_ordner = r"/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/laser_dark"
result = calibration_functions.get_laserline_points(laser_dark_ordner, kamera_parameters["mtx"], kamera_parameters["dist"], result)
Plane_pointsR = []
Plane_pointsL = []
for i,j in enumerate(result):
    if j["points_l"] is not None:
        for (u,v) in j["points_l"]:
            cam_vec = calibration_functions.pixel_to_cameraVector(u, v, kamera_parameters["mtx"])
            normal = calibration_functions.get_normal_Checkerboard(j["rotation"])
            P = calibration_functions.get_intersection(cam_vec, j["translation"], normal)
            Plane_pointsL.append(P)                                                                                                       

    if j["points_r"] is not None:
        for (u, v) in j["points_r"]:
            cam_vec = calibration_functions.pixel_to_cameraVector(u, v, kamera_parameters["mtx"])
            normal = calibration_functions.get_normal_Checkerboard(j["rotation"])
            P = calibration_functions.get_intersection(cam_vec, j["translation"], normal)
            Plane_pointsR.append(P)
# --- Punkte vorbereiten ---
scanned_PointsR = np.asarray(Plane_pointsR, dtype=float).reshape(-1, 3)
scanned_PointsL = np.asarray(Plane_pointsL, dtype=float).reshape(-1, 3)

# --- Ebene fitten ---
centroid_L, normal_L = calibration_functions.fit_plane_svd(scanned_PointsL)
centroid_R, normal_R = calibration_functions.fit_plane_svd(scanned_PointsR)


# === Ergebnisse speichern ===
save_path = r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_data_03.npz"

np.savez(
    save_path,
    mtx=kamera_parameters["mtx"],
    dist=kamera_parameters["dist"],
    centroid_L=centroid_L,
    normal_L=normal_L,
    centroid_R=centroid_R,
    normal_R=normal_R
)

print(f"Kalibrierungsdaten gespeichert unter:\n{save_path}")
