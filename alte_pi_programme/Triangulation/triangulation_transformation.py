import numpy as np

def transform_vector(vec, rotvec_CW, trvec_CW):
    return 1
def get_transformation_data(base_drehteller, rotvec_drehteller):
    rotvec_CW = rotvec_drehteller / np.linalg.norm(rotvec_drehteller)
    rotvec_CW = np.asarray(rotvec_CW, dtype = float).reshape(3)

def make_world_from_circle(center_c, basis):
    """
    center_c :  (3,)   Kreismittelpunkt im Kamera-KS
    basis    : (p0, u, v) aus fit_circle_3d()
    
    Weltbasis = (u, v, n)
    R_wc = Welt-zu-Kamera-Rotation
    t_wc = Welt-zu-Kamera-Translation
    """
    p0, u, v = basis
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    n = np.cross(u, v); n /= np.linalg.norm(n)

    # Spalten = Weltachsen IM Kamera-KS
    R_wc = np.column_stack((u, v, n))   # 3x3
    t_wc = -R_wc @ center_c.reshape(3,) # 3x1

    return R_wc, t_wc

def camtoworld(P_c, R_wc, t_wc):
                # Welt-Ursprung (Kamera-Raum-Koordinaten)

    # Transform: Kamera → Welt: p_w = R_w^T (p_c - origin)
    return (R_wc.T @ (P_c - t_wc).T).T




