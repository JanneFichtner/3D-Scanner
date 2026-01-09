'''
Bei der kalibrierung wird:
    1. Kameramatrix ermittelt
    2. Laserebenen ermittelt
    3. Achse des Drehtellers ermittelt

Hardware ablauf:
    -Drehteller dreht sich und alle 5° checke ich, ob schachbrett erkannt wird
    -winkelbereiche werden defieniert
    -innerhalb der winkelbereiche bilder aufnehmen:
        -laser dark
        -ohne laser hell
        -alles in arrays speichern ohne auf sd karte zu schreiben
    -mit den bildern kann man 1. 2. 3. kalibrieren
Software ablauf:
    -...
'''
import time
import cv2
import numpy as np
class Kalibrierung():
    def __init__(self, nano, cameras, encoder):
        #referenzen der hardware objekte speichern
        self.nano = nano
        self.cameras = cameras
        self.encoder = encoder
        
        #arays für die kalibrierungsbilder
        self.images_hell = []
        self.images_laser = []
    
    def get_calibration_data(self):
        agle_range = self.get_angle_range()
        angle = self.encoder.get_angle()
        
        import numpy as np

        for i in np.arange(0, 90, 5):
            grad = i % 360
            
            angle = self.encoder.get_angle()
            self.nano.move_to_angle(angle,grad)
            time.sleep(2)
            self.images_hell.append(self.cameras.take_pic())

            self.nano.send_data(5,1)
            time.sleep(0.25)
            self.images_laser.append(self.cameras.take_pic())
            self.nano.send_data(6,1)
        import cv2
        import os

        # Nach der Schleife:
        path = "/home/janne/Desktop/bilder"
        os.makedirs(path, exist_ok=True)

        
        for i, img in enumerate(self.images_hell):
            # Konvertierung von RGB (Kamera) nach BGR (OpenCV) und Speichern
            full_path = f"{path}/hell_{i:02d}.jpg"
            cv2.imwrite(full_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"{len(self.images_hell)} Bilder gespeichert.")

        self.checkerboard_calibration(self.images_hell)
    def get_angle_range(self):
        return([270, 90])
    
    def checkerboard_calibration(
                self,
                images, 
                square_size = 4.444, 
                CHECKERBOARD = (5,8)
                ):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objpoints, imgpoints = [], []
            valid_size = None  # 1. Initialisierung hinzufügen
            
            objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
        
            for img in images:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                b, g, r = cv2.split(img_bgr)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(g)

                ret, corners = cv2.findChessboardCorners(
                    gray, CHECKERBOARD,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if ret:
                    print("Schachbrett gefunden!")
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    valid_size = gray.shape[::-1]
                else:
                    print("Schachbrett NICHT gefunden.")

            # 2. Check: Wurden überhaupt Punkte gefunden?
            if len(objpoints) > 0 and valid_size is not None:
                # Kalibrierung außerhalb des Loops aufrufen
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, valid_size, None, None
                )
                
                result = []
                for rvec, tvec in zip(rvecs, tvecs):
                    result.append({
                        "translation": tvec,
                        "rotation": rvec,
                        "shape": valid_size
                    })
                
                self.kamera_parameters = {
                    "mtx": mtx,
                    "dist": dist
                }
                print("Kalibrierung erfolgreich!")
                return result, self.kamera_parameters
            else:
                print("Fehler: Es konnten nicht genügend Schachbrett-Ecken für eine Kalibrierung gefunden werden.")
                return None, None
            