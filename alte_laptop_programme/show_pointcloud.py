#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from pathlib import Path

PATH = r"G:\Meine Ablage\Studium\Master\3. Semester\Masterprojekt\Punktwolken_Scans_Prototyp\Maus_23102025.npz"

def load_points(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    pts = data["P"] if "P" in data else data[list(data.files)[0]]
    return np.asarray(pts, dtype=float).reshape(-1, 3)

def show_open3d(points: np.ndarray):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # Optional: Farbe setzen
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.2, 0.6, 1.0])
    o3d.visualization.draw_geometries([pcd])  # interaktive Ansicht

if __name__ == "__main__":
    pts = load_points(PATH)
    show_open3d(pts)
