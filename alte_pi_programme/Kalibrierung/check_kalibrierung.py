import numpy as np
import cv2

data = np.load(r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_data_03.npz")
mtx = data["mtx"]
dist = data["dist"]
centroid_L = data["centroid_L"]
normal_L = data["normal_L"]
centroid_R = data["centroid_R"]
normal_R = data["normal_R"]

#laden der drehteller daten
data_drehteller = np.load("/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_drehteller.npz")

base_drehteller = data_drehteller["base_drehteller"]
normal_drehteller = data_drehteller["normal_drehteller"]
radius_drehteller = data_drehteller["radius"]
punkte_drehteller = data_drehteller["pts"]
rotvecs = data_drehteller["rotvecs"]

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_circle_3d(points,
                   center3d,
                   normal,
                   radius,
                   rotvecs,
                   plot_normals=True,
                   plane_basis=None,
                   axis_scale=1.2,
                   laser_npz_path=r"/home/janne/Desktop/Masterprojekt/Kalibrierung/calibration_data_03.npz",
                   laser_alpha=0.25):
    """
    points      : (N,3) Originalpunkte (Kamera-KS)
    center3d    : Mittelpunkt des Kreises in 3D
    normal      : Ebenennormalenvektor (Einheitslänge)
    radius      : Kreisradius
    rotvecs     : (N,3) Rotationsvektoren zu den Punkten (für Normallen-Zeichnung)
    plot_normals: True => Normale an jedem Punkt zeichnen
    plane_basis : (p0,u,v) aus fit_circle_3d() (optional, aber empfohlen)
    axis_scale  : Darstellungslimit im Plot
    laser_npz_path : Pfad zur NPZ mit Laser-Ebenen (erwartete Keys: centroid_L, normal_L, centroid_R, normal_R)
    laser_alpha : Transparenz der Laserflächen
    """

    def _unit(x, eps=1e-12):
        n = np.linalg.norm(x)
        return x / (n if n > eps else eps)

    # --- lokale Basis der Drehteller-Ebene ---
    if plane_basis is not None:
        p0, u, v = plane_basis
        n_fit = _unit(normal)
    else:
        n_fit = _unit(normal)
        a = np.array([1.0, 0.0, 0.0]) if abs(n_fit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = _unit(np.cross(n_fit, a))
        v = np.cross(n_fit, u)

    # --- Kreis generieren (in 3D) ---
    t = np.linspace(0, 2*np.pi, 200)
    circle3d = center3d + radius*np.outer(np.cos(t), u) + radius*np.outer(np.sin(t), v)

    # --- Rotationsachse als Linie ---
    axis_len = radius * axis_scale
    A = center3d - axis_len*n_fit
    B = center3d + axis_len*n_fit

    # --- Plot vorbereiten ---
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Punkte
    P = np.asarray(points)
    ax.scatter(P[:,0], P[:,1], P[:,2], s=25, color='blue', label="Messpunkte (Drehteller)")

    # Kreis
    ax.plot(circle3d[:,0], circle3d[:,1], circle3d[:,2], color='red', label="Gefitteter Kreis (Drehteller)")

    # Mittelpunkt
    ax.scatter(center3d[0], center3d[1], center3d[2], color='green', s=80, label="Kreismittelpunkt")

    # Rotationsachse
    ax.plot([A[0],B[0]],[A[1],B[1]],[A[2],B[2]], color='black', linewidth=2, label="Rotationsachse")

    # optionale Normallen an den Messpunkten (aus rotvecs)
    if plot_normals and rotvecs is not None:
        for point, rvec in zip(points, rotvecs):
            s = point
            R, _ = cv2.Rodrigues(rvec)
            n_board = R[:, 2]   # z-Achse des Boards → Normale im Kamera-KS
            e = s + n_board * (0.2 * axis_len)  # Pfeillänge moderat
            ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]], color='blue', linewidth=1)

    # ==========================
    # Laser-Ebenen laden + plotten
    # ==========================
    plane_meshes = []   # zum gemeinsamen Scaling
    try:
        data_laser = np.load(laser_npz_path)
        # Robust: Keys aus 01/03 kompatibel
        centroid_L = data_laser.get("centroid_L")
        normal_L   = data_laser.get("normal_L")
        centroid_R = data_laser.get("centroid_R")
        normal_R   = data_laser.get("normal_R")

        if centroid_L is None or normal_L is None or centroid_R is None or normal_R is None:
            raise KeyError("Erwarte Keys 'centroid_L', 'normal_L', 'centroid_R', 'normal_R' im Laser-NPZ.")

        centroid_L = np.asarray(centroid_L).reshape(3)
        normal_L   = _unit(np.asarray(normal_L).reshape(3))
        centroid_R = np.asarray(centroid_R).reshape(3)
        normal_R   = _unit(np.asarray(normal_R).reshape(3))

        # Hilfsfunktion: aus (c, n) ein Planquadrat als Mesh erzeugen
        def _plane_mesh(c, n, scale):
            # stabile Tangentialbasis (u_p, v_p)
            a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u_p = _unit(np.cross(n, a))
            v_p = np.cross(n, u_p)
            # Quadrat-Seitenlänge: 2*scale
            corners = np.array([
                c + scale*(+u_p + v_p),
                c + scale*(+u_p - v_p),
                c + scale*(-u_p - v_p),
                c + scale*(-u_p + v_p),
            ])  # (4,3)
            # für ein „gefülltes“ Rechteck Triangulieren
            tri1 = [corners[0], corners[1], corners[2]]
            tri2 = [corners[0], corners[2], corners[3]]
            return np.array([tri1, tri2])  # (2,3,3)

        # Größe der Ebenen am Drehteller ausrichten
        plane_scale = max(radius * 3, 0.5)  # etwas größer als der Kreis

        mesh_L = _plane_mesh(centroid_L, normal_L, plane_scale)
        mesh_R = _plane_mesh(centroid_R, normal_R, plane_scale)

        # Plotten: L rot, R blau (semi-transparent)
        coll_L = Poly3DCollection(mesh_L, facecolors='red', alpha=laser_alpha, edgecolors='none')
        coll_R = Poly3DCollection(mesh_R, facecolors='blue', alpha=laser_alpha, edgecolors='none')
        ax.add_collection3d(coll_L)
        ax.add_collection3d(coll_R)

        # Normalenpfeile der Laser (kurz)
        l_len = 0.5 * plane_scale
        ax.plot([centroid_L[0], centroid_L[0]+normal_L[0]*l_len],
                [centroid_L[1], centroid_L[1]+normal_L[1]*l_len],
                [centroid_L[2], centroid_L[2]+normal_L[2]*l_len],
                color='red', linewidth=2)
        ax.plot([centroid_R[0], centroid_R[0]+normal_R[0]*l_len],
                [centroid_R[1], centroid_R[1]+normal_R[1]*l_len],
                [centroid_R[2], centroid_R[2]+normal_R[2]*l_len],
                color='blue', linewidth=2)

        # Dummy Artists für Legende (damit Flächen einen Eintrag bekommen)
        from matplotlib.lines import Line2D
        proxy_L = Line2D([0],[0], linestyle="none", marker="s", markersize=10, markerfacecolor="red", alpha=laser_alpha, label="Laser-Ebene L (rot)")
        proxy_R = Line2D([0],[0], linestyle="none", marker="s", markersize=10, markerfacecolor="blue", alpha=laser_alpha, label="Laser-Ebene R (blau)")
        ax.add_artist(proxy_L); ax.add_artist(proxy_R)

        # Für Scaling sammeln
        plane_meshes = [mesh_L.reshape(-1,3), mesh_R.reshape(-1,3)]
    except Exception as e:
        print(f"[Hinweis] Laser-Ebenen konnten nicht geladen/gezeichnet werden ({e}). Fahre ohne fort.")

    # Achsenbeschriftung
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drehteller + Laser-Ebenen")

    # Legende
    ax.legend(loc='upper right')

    # *** GLEICHE ACHSSKALIERUNG + cubic aspect ***
    stacks = [P, circle3d, center3d.reshape(1,3), A.reshape(1,3), B.reshape(1,3)]
    for m in plane_meshes:
        stacks.append(m)
    all_pts = np.vstack(stacks)
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

    # Erzwingt optisches 1:1:1
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


plot_circle_3d(punkte_drehteller, base_drehteller,normal_drehteller,radius_drehteller, rotvecs)
