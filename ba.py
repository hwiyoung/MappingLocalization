import Metashape
import time
from tabulate import tabulate
from matches_to_xml import *
import numpy as np


class BA_api:

    def __init__(self):
        print("InnoPAM")

    def alignphotos_1(self, image_path, region):

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

    def alignphotos_2(self, image_path, region):

        start_time = time.time()

        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(image_path)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for train_R images
        chunk.addSensor()
        for i in range(int((len(chunk.cameras)-2)/2)):
            chunk.cameras[2*i+1].sensor = chunk.sensors[-1]

        chunk.sensors[1].label = "train_R"
        chunk.sensors[1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[1].width = chunk.sensors[0].width
        chunk.sensors[1].height = chunk.sensors[0].height
        chunk.sensors[1].focal_length = chunk.sensors[0].focal_length

        ref_calib_R = Metashape.Calibration()
        ref_calib_R.load('calibration/calib_rcam.xml')
        chunk.sensors[1].user_calib = ref_calib_R
        chunk.sensors[1].fixed_calibration = True

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
        ref_calib_L = Metashape.Calibration()
        ref_calib_L.load('calibration/calib_lcam.xml')
        chunk.sensors[0].user_calib = ref_calib_L
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_all_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(image_path)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # print("==save project=================================================")
        # path = "./test.psz"
        # doc.save(path)
        # print("===============================================================")

        # Match photos
        chunk.matchPhotos(downscale=2, keep_keypoints=True)

        # Align cameras
        chunk.alignCameras(adaptive_fitting=True)

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

    def alignphotos_3(self, images_train, images_test_L, images_test_R, number_of_images, region):
        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(images_train)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for train_R images
        chunk.addSensor()
        chunk.sensors[-1] = chunk.sensors[0]
        for i in range(int(len(chunk.cameras)/2)):
            chunk.cameras[2*i+1].sensor = chunk.sensors[-1]

        chunk.sensors[-1].label = "train_R"
        chunk.sensors[-1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[-1].width = chunk.sensors[0].width
        chunk.sensors[-1].height = chunk.sensors[0].height
        chunk.sensors[-1].focal_length = chunk.sensors[0].focal_length
        chunk.sensors[-1].bands = ['Red', 'Green', 'Blue']

        ref_calib_R = Metashape.Calibration()
        ref_calib_R.load('calibration/calib_rcam2.xml')
        chunk.sensors[-1].user_calib = ref_calib_R
        chunk.sensors[-1].fixed_calibration = True

        # Import pre-calibrated IO into the sensor of the reference images
        ref_calib_L = Metashape.Calibration()
        ref_calib_L.load('calibration/calib_lcam2.xml')
        chunk.sensors[0].user_calib = ref_calib_L
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_all_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(images_train)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # Add a chunk for test L images
        chunk2 = doc.addChunk()
        chunk2.addPhotos(images_test_L)

        chunk2.sensors[0].label = "test_L"
        chunk2.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk2.sensors[0].width = chunk.sensors[0].width
        chunk2.sensors[0].height = chunk.sensors[0].height
        chunk2.sensors[0].focal_length = chunk.sensors[0].focal_length

        # Add a chunk for test R images
        chunk3 = doc.addChunk()
        chunk3.addPhotos(images_test_R)

        chunk3.sensors[0].label = "test_R"
        chunk3.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk3.sensors[0].width = chunk.sensors[0].width
        chunk3.sensors[0].height = chunk.sensors[0].height
        chunk3.sensors[0].focal_length = chunk.sensors[0].focal_length

        doc.mergeChunks(chunks=[0, 1, 2])
        chunk_to_process = doc.chunks[-1]

        chunk_to_process.marker_projection_accuracy = 0.1

        # Match photos
        # chunk.matchPhotos(downscale=2, keep_keypoints=True)     # Medium
        chunk_to_process.matchPhotos(keep_keypoints=True)  # High

        print("==save project=================================================")
        path = "./" + images_test_L[0].split("\\")[1] + "_" + str(number_of_images) + ".psz"
        doc.save(path)
        print("===============================================================")

        # Align cameras
        # chunk.alignCameras(adaptive_fitting=True)
        chunk_to_process.alignCameras()

        # chunk_to_process.optimizeCameras()



        camera_L = chunk_to_process.cameras[-2]
        camera_R = chunk_to_process.cameras[-1]
        cameras = [camera_L, camera_R]
        print(cameras)

        EOs = ["" for i in range(2)]
        if not camera_L.transform or not camera_R.transform or not camera_L.center or not camera_R.center:
            print("=======================================")
            print("|| There is no transformation matrix ||")
            print("=======================================")
            return camera_L, EOs[0], camera_R, EOs[1]

        i = 0
        for camera in cameras:
            estimated_coord = chunk_to_process.crs.project(
                chunk_to_process.transform.matrix.mulp(camera.center))  # estimated XYZ in coordinate system units
            T = chunk_to_process.transform.matrix
            m = chunk_to_process.crs.localframe(
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

    def alignphotos_3_1(self, images_train, images_test_L, images_test_R, number_of_images, sequence, idx_test, region):
        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(images_train)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for train_R images
        chunk.addSensor()
        chunk.sensors[-1] = chunk.sensors[0]
        for i in range(int(len(chunk.cameras)/2)):
            chunk.cameras[2*i+1].sensor = chunk.sensors[-1]

        chunk.sensors[-1].label = "train_R"
        chunk.sensors[-1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[-1].width = chunk.sensors[0].width
        chunk.sensors[-1].height = chunk.sensors[0].height
        chunk.sensors[-1].focal_length = chunk.sensors[0].focal_length
        chunk.sensors[-1].bands = ['Red', 'Green', 'Blue']

        ref_calib_R = Metashape.Calibration()
        ref_calib_R.load('calibration/calib_rcam2.xml')
        chunk.sensors[-1].user_calib = ref_calib_R
        chunk.sensors[-1].fixed_calibration = True

        # Import pre-calibrated IO into the sensor of the reference images
        ref_calib_L = Metashape.Calibration()
        ref_calib_L.load('calibration/calib_lcam2.xml')
        chunk.sensors[0].user_calib = ref_calib_L
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_all_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(images_train)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # Add a chunk for test L images
        chunk2 = doc.addChunk()
        chunk2.addPhotos(images_test_L)

        chunk2.sensors[0].label = "test_L"
        chunk2.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk2.sensors[0].width = chunk.sensors[0].width
        chunk2.sensors[0].height = chunk.sensors[0].height
        chunk2.sensors[0].focal_length = chunk.sensors[0].focal_length

        # Add a chunk for test R images
        chunk3 = doc.addChunk()
        chunk3.addPhotos(images_test_R)

        chunk3.sensors[0].label = "test_R"
        chunk3.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk3.sensors[0].width = chunk.sensors[0].width
        chunk3.sensors[0].height = chunk.sensors[0].height
        chunk3.sensors[0].focal_length = chunk.sensors[0].focal_length

        doc.mergeChunks(chunks=[0, 1, 2])
        chunk_to_process = doc.chunks[-1]

        chunk_to_process.marker_projection_accuracy = 0.1

        chunk_to_process.exportCameras("cameras.xml")

        # test_matches = np.array([[11, 12, 13, 14, 15, 16, 17, 18],
        #                          [21, 22, 23, 24, 25, 26, 27, 28],
        #                          [31, 32, 33, 34, 35, 36, 37, 38]])
        # write_to_xml("cameras.xml", test_matches, sequence)

        match_path = "superglue/" + region + str(idx_test).zfill(2) + ".csv"
        matches = np.loadtxt(match_path, dtype=np.int, delimiter=",")
        write_to_xml("cameras.xml", matches, sequence)

        # Import XML
        chunk_to_process.importMarkers("markers.xml")

        # Match photos
        # chunk.matchPhotos(downscale=2, keep_keypoints=True)     # Medium
        chunk_to_process.matchPhotos(keep_keypoints=True)  # High

        # Align cameras
        # chunk.alignCameras(adaptive_fitting=True)
        chunk_to_process.alignCameras()

        # chunk_to_process.optimizeCameras()

        print("==save project=================================================")
        path = "./" + images_test_L[0].split("\\")[1] + "_" + str(number_of_images) + "_1.psz"
        doc.save(path)
        print("===============================================================")

        camera_L = chunk_to_process.cameras[-2]
        camera_R = chunk_to_process.cameras[-1]
        cameras = [camera_L, camera_R]
        print(cameras)

        EOs = ["" for i in range(2)]
        if not camera_L.transform or not camera_R.transform or not camera_L.center or not camera_R.center:
            print("=======================================")
            print("|| There is no transformation matrix ||")
            print("=======================================")
            return camera_L, EOs[0], camera_R, EOs[1]

        i = 0
        for camera in cameras:
            estimated_coord = chunk_to_process.crs.project(
                chunk_to_process.transform.matrix.mulp(camera.center))  # estimated XYZ in coordinate system units
            T = chunk_to_process.transform.matrix
            m = chunk_to_process.crs.localframe(
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

    def alignphotos_3_2(self, images_train, images_test_L, images_test_R, number_of_images, sequence, region):
        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(images_train)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for train_R images
        chunk.addSensor()
        chunk.sensors[-1] = chunk.sensors[0]
        for i in range(int(len(chunk.cameras)/2)):
            chunk.cameras[2*i+1].sensor = chunk.sensors[-1]

        chunk.sensors[-1].label = "train_R"
        chunk.sensors[-1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[-1].width = chunk.sensors[0].width
        chunk.sensors[-1].height = chunk.sensors[0].height
        chunk.sensors[-1].focal_length = chunk.sensors[0].focal_length
        chunk.sensors[-1].bands = ['Red', 'Green', 'Blue']

        ref_calib_R = Metashape.Calibration()
        ref_calib_R.load('calibration/calib_rcam2.xml')
        chunk.sensors[-1].user_calib = ref_calib_R
        chunk.sensors[-1].fixed_calibration = True

        # Import pre-calibrated IO into the sensor of the reference images
        ref_calib_L = Metashape.Calibration()
        ref_calib_L.load('calibration/calib_lcam2.xml')
        chunk.sensors[0].user_calib = ref_calib_L
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_all_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(images_train)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # Add a chunk for test L images
        chunk2 = doc.addChunk()
        chunk2.addPhotos(images_test_L)

        chunk2.sensors[0].label = "test_L"
        chunk2.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk2.sensors[0].width = chunk.sensors[0].width
        chunk2.sensors[0].height = chunk.sensors[0].height
        chunk2.sensors[0].focal_length = chunk.sensors[0].focal_length

        # Add a chunk for test R images
        chunk3 = doc.addChunk()
        chunk3.addPhotos(images_test_R)

        chunk3.sensors[0].label = "test_R"
        chunk3.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk3.sensors[0].width = chunk.sensors[0].width
        chunk3.sensors[0].height = chunk.sensors[0].height
        chunk3.sensors[0].focal_length = chunk.sensors[0].focal_length

        doc.mergeChunks(chunks=[0, 1, 2])
        chunk_to_process = doc.chunks[-1]

        chunk_to_process.marker_projection_accuracy = 0.1

        # Import EO of the netvlad train image to the last test image
        # 49_L
        chunk_to_process.cameras[-2].reference.location = chunk_to_process.cameras[
                                                            sequence * 2].reference.location
        chunk_to_process.cameras[-2].reference.rotation = chunk_to_process.cameras[
                                                            sequence * 2].reference.rotation
        chunk_to_process.cameras[-2].reference.location_accuracy = Metashape.Vector([10, 10, 10])
        chunk_to_process.cameras[-2].reference.rotation_accuracy = Metashape.Vector([10, 10, 10])
        chunk_to_process.cameras[-2].reference.rotation_enabled = True
        # 49_R
        chunk_to_process.cameras[-1].reference.location = chunk_to_process.cameras[
                                                            sequence * 2 + 1].reference.location
        chunk_to_process.cameras[-1].reference.rotation = chunk_to_process.cameras[
                                                            sequence * 2 + 1].reference.rotation
        chunk_to_process.cameras[-1].reference.location_accuracy = Metashape.Vector([10, 10, 10])
        chunk_to_process.cameras[-1].reference.rotation_accuracy = Metashape.Vector([10, 10, 10])
        chunk_to_process.cameras[-1].reference.rotation_enabled = True

        # Match photos
        # chunk.matchPhotos(downscale=2, keep_keypoints=True)     # Medium
        chunk_to_process.matchPhotos(keep_keypoints=True)  # High

        # Align cameras
        # chunk.alignCameras(adaptive_fitting=True)
        chunk_to_process.alignCameras()

        # chunk_to_process.optimizeCameras()

        print("==save project=================================================")
        path = "./" + images_test_L[0].split("\\")[1] + "_" + str(number_of_images) + ".psz"
        doc.save(path)
        print("===============================================================")

        camera_L = chunk_to_process.cameras[-2]
        camera_R = chunk_to_process.cameras[-1]
        cameras = [camera_L, camera_R]
        print(cameras)

        EOs = ["" for i in range(2)]
        if not camera_L.transform or not camera_R.transform or not camera_L.center or not camera_R.center:
            print("=======================================")
            print("|| There is no transformation matrix ||")
            print("=======================================")
            return camera_L, EOs[0], camera_R, EOs[1]

        i = 0
        for camera in cameras:
            estimated_coord = chunk_to_process.crs.project(
                chunk_to_process.transform.matrix.mulp(camera.center))  # estimated XYZ in coordinate system units
            T = chunk_to_process.transform.matrix
            m = chunk_to_process.crs.localframe(
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

    def alignphotos_3_3(self, images_train, images_test_L, images_test_R, number_of_images, sequence, idx_test, region):
        doc = Metashape.Document()
        chunk = doc.addChunk()
        chunk.addPhotos(images_train)

        # Set IO
        chunk.sensors[0].pixel_size = Metashape.Vector([0.00645, 0.00645])
        chunk.sensors[0].focal_length = 8.25

        # Add a sensor group for train_R images
        chunk.addSensor()
        chunk.sensors[-1] = chunk.sensors[0]
        for i in range(int(len(chunk.cameras)/2)):
            chunk.cameras[2*i+1].sensor = chunk.sensors[-1]

        chunk.sensors[-1].label = "train_R"
        chunk.sensors[-1].pixel_size = chunk.sensors[0].pixel_size
        chunk.sensors[-1].width = chunk.sensors[0].width
        chunk.sensors[-1].height = chunk.sensors[0].height
        chunk.sensors[-1].focal_length = chunk.sensors[0].focal_length
        chunk.sensors[-1].bands = ['Red', 'Green', 'Blue']

        ref_calib_R = Metashape.Calibration()
        ref_calib_R.load('calibration/calib_rcam2.xml')
        chunk.sensors[-1].user_calib = ref_calib_R
        chunk.sensors[-1].fixed_calibration = True

        # Import pre-calibrated IO into the sensor of the reference images
        ref_calib_L = Metashape.Calibration()
        ref_calib_L.load('calibration/calib_lcam2.xml')
        chunk.sensors[0].user_calib = ref_calib_L
        chunk.sensors[0].fixed_calibration = True
        chunk.sensors[0].label = "train_L"

        # Import pose from poses.txt
        chunk.importReference(region + "_all_poses.csv", Metashape.ReferenceFormatCSV, "nxyzabc", ",", skip_rows=1)
        chunk.camera_location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        chunk.camera_rotation_accuracy = Metashape.Vector([0.01, 0.01, 0.01])
        for i in range(len(images_train)):
            chunk.cameras[i].reference.location_enabled = True
            chunk.cameras[i].reference.rotation_enabled = True

        # Add a chunk for test L images
        chunk2 = doc.addChunk()
        chunk2.addPhotos(images_test_L)

        chunk2.sensors[0].label = "test_L"
        chunk2.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk2.sensors[0].width = chunk.sensors[0].width
        chunk2.sensors[0].height = chunk.sensors[0].height
        chunk2.sensors[0].focal_length = chunk.sensors[0].focal_length

        # Add a chunk for test R images
        chunk3 = doc.addChunk()
        chunk3.addPhotos(images_test_R)

        chunk3.sensors[0].label = "test_R"
        chunk3.sensors[0].pixel_size = chunk.sensors[0].pixel_size
        chunk3.sensors[0].width = chunk.sensors[0].width
        chunk3.sensors[0].height = chunk.sensors[0].height
        chunk3.sensors[0].focal_length = chunk.sensors[0].focal_length

        doc.mergeChunks(chunks=[0, 1, 2])
        chunk_to_process = doc.chunks[-1]

        chunk_to_process.marker_projection_accuracy = 0.1

        chunk_to_process.exportCameras("cameras.xml")

        # test_matches = np.array([[11, 12, 13, 14, 15, 16, 17, 18],
        #                          [21, 22, 23, 24, 25, 26, 27, 28],
        #                          [31, 32, 33, 34, 35, 36, 37, 38]])
        # write_to_xml("cameras.xml", test_matches, sequence)

        match_path = "superglue/" + region + str(idx_test).zfill(2) + ".csv"
        matches = np.loadtxt(match_path, dtype=np.int, delimiter=",")
        write_to_xml2("cameras.xml", matches, sequence)

        # Import XML
        chunk_to_process.importMarkers("markers.xml")

        # Match photos
        # chunk.matchPhotos(downscale=2, keep_keypoints=True)     # Medium
        chunk_to_process.matchPhotos(keep_keypoints=True)  # High

        # Align cameras
        # chunk.alignCameras(adaptive_fitting=True)
        chunk_to_process.alignCameras()

        # chunk_to_process.optimizeCameras()

        print("==save project=================================================")
        path = "./" + images_test_L[0].split("\\")[1] + "_" + str(number_of_images) + "_1.psz"
        doc.save(path)
        print("===============================================================")

        camera_L = chunk_to_process.cameras[-2]
        camera_R = chunk_to_process.cameras[-1]
        cameras = [camera_L, camera_R]
        print(cameras)

        EOs = ["" for i in range(2)]
        if not camera_L.transform or not camera_R.transform or not camera_L.center or not camera_R.center:
            print("=======================================")
            print("|| There is no transformation matrix ||")
            print("=======================================")
            return camera_L, EOs[0], camera_R, EOs[1]

        i = 0
        for camera in cameras:
            estimated_coord = chunk_to_process.crs.project(
                chunk_to_process.transform.matrix.mulp(camera.center))  # estimated XYZ in coordinate system units
            T = chunk_to_process.transform.matrix
            m = chunk_to_process.crs.localframe(
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