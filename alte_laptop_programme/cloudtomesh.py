import numpy as np
import open3d as o3d
import os

# <<< Pfad zu deiner Datei >>>
PATH = r"G:\Meine Ablage\Studium\Master\3. Semester\Masterprojekt\Punktwolken_Scans_Prototyp\Maus_23102025.npz"
OUTPUT = "spiral_mesh"


# --- Punktwolke laden ---
data = np.load(PATH)
if "P" in data:
    points = data["P"]
else:
    # Fallback: erstes Array nehmen, falls Name anders ist
    points = list(data.values())[0]

points = points[:, :3].astype(np.float32)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

print(f"Punktwolke geladen: {points.shape[0]} Punkte")

# --- normalisieren / Normals schätzen ---
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50)
)

# --- Mesh erzeugen (Poisson) ---
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# bounding box crop (gegen Poisson-Artefakte)
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

mesh.compute_vertex_normals()

# --- Mesh als STL und PLY speichern ---
o3d.io.write_triangle_mesh(OUTPUT + ".stl", mesh)
o3d.io.write_triangle_mesh(OUTPUT + ".ply", mesh)

print(f"Mesh gespeichert als: {OUTPUT}.stl und {OUTPUT}.ply")

# --- optional anzeigen ---
o3d.visualization.draw_geometries([mesh])
