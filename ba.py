import Metashape
import time
from tabulate import tabulate

class BA_api:

    def __init__(self):
        print("InnoPAM")

    def alignphotos(self, image_path, region):

        start_time = time.time()

        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(image_path)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for test images
        chunk.addSensor()
        chunk.cameras[-2].sensor = chunk.sensors[-1]
        chunk.cameras[-1].sensor = chunk.sensors[-1]

        chunk.sensors[-1].label = "test"
        chunk.sensors[-1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[-1].width = chunk.sensors[0].width
        chunk.sensors[-1].height = chunk.sensors[0].height
        chunk.sensors[-1].focal_length = chunk.sensors[0].focal_length

        # Import pre-calibrated IO into the sensor of the reference images
        ref_calib = Metashape.Calibration()
        ref_calib.load('calibration/calib_lcam.xml')
        chunk.sensors[0].user_calib = ref_calib
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_lcam_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(image_path)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # Match photos
        chunk.matchPhotos(downscale=2, keep_keypoints=True)

        # Align cameras
        chunk.alignCameras(adaptive_fitting=True)

        # print("==save project=================================================")
        # path = "./test.psz"
        # doc.save(path)
        # print("===============================================================")

        # chunk.optimizeCameras()

        camera_L = chunk.cameras[-2]
        camera_R = chunk.cameras[-1]
        cameras = [camera_L, camera_R]

        EOs = ["" for i in range(2)]
        if not camera_L.transform or not camera_R.transform or not camera_L.center or not camera_R.center:
            print("=======================================")
            print("|| There is no transformation matrix ||")
            print("=======================================")
            return camera_L, EOs[0], camera_R, EOs[1]

        i = 0
        for camera in cameras:
            estimated_coord = chunk.crs.project(
                chunk.transform.matrix.mulp(camera.center))  # estimated XYZ in coordinate system units
            T = chunk.transform.matrix
            m = chunk.crs.localframe(
                T.mulp(camera.center))  # transformation matrix to the LSE coordinates in the given point
            R = (m * T * camera.transform * Metashape.Matrix().Diag([1, -1, -1, 1])).rotation()
            estimated_ypr = Metashape.utils.mat2ypr(R)  # estimated orientation angles - yaw, pitch, roll
            estimated_opk = Metashape.utils.mat2opk(R)  # estimated orientation angles - omega, phi, kappa

            pos = list(estimated_coord)
            ypr = list(estimated_ypr)
            opk = list(estimated_opk)
            eo = [pos[0], pos[1], pos[2], ypr[0], ypr[1], ypr[2], opk[0], opk[1], opk[2]]
            print(tabulate([[eo[0], eo[1], eo[2], eo[3], eo[4], eo[5], eo[6], eo[7], eo[8]]],
                           headers=["X(m)", "Y(m)", "Z(m)", "Yaw(deg)", "Pitch(deg)", "Roll(deg)"
                                    , "Omega(deg)", "Phi(deg)", "Kappa(deg)"],
                           tablefmt='psql'))
            EOs[i] = eo
            i += 1

        return camera_L, EOs[0], camera_R, EOs[1]
