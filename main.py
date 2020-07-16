import numpy as np
import os
from ba import BA_api

regions = ["yeouido", "pangyo"]
for region in regions:
    f = open(region + "_L.txt", "w")
    f.write("Label_L" + "\t" + "X(m)" + "\t" + "Y(m)" + "\t" + "Z(m)" + "\t"
            + "Yaw(deg)" + "\t" + "Pitch(deg)" + "\t" + "Roll(deg)" + "\t"
            + "Omega(deg)" + "\t" + "Phi(deg)" + "\t" + "Kappa(deg)" + "\t"
            + "Label_R" + "\t" + "X(m)" + "\t" + "Y(m)" + "\t" + "Z(m)" + "\t"
            + "Yaw(deg)" + "\t" + "Pitch(deg)" + "\t" + "Roll(deg)" + "\t"
            + "Omega(deg)" + "\t" + "Phi(deg)" + "\t" + "Kappa(deg)" + "\n")

    # 1. Load a csv file
    reader = np.loadtxt("netvlad/retrieved naverlabs " + region + " image indices.csv", dtype=np.int, delimiter=",")

    # 2. Match the closest db images corresponding to query images
    query_imgs = reader[:, 0]
    db_imgs = reader[:, 1]

    query_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/test"
    query_L_path = ["" for i in range(len(query_imgs))]
    query_R_path = ["" for i in range(len(query_imgs))]
    for idx_q in range(len(query_imgs)):
        query_L_idx = region + str(idx_q).zfill(2) + "/049_L.png"
        query_R_idx = region + str(idx_q).zfill(2) + "/049_R.png"
        query_L_path[idx_q] = os.path.join(query_dir, query_L_idx)
        query_R_path[idx_q] = os.path.join(query_dir, query_R_idx)

        # 3. Select db images using result of step 2
        # 3-1. Select front & back 50 db images of result of step 2
        db_L_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/train/images/left"
        db_R_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/train/images/right"
        seq = 25
        i = 0
        # images_to_process = ["" for i in range(seq * 2 * 2)]
        images_to_process = ["" for i in range(seq * 2)]    # test
        for idx_d in range(db_imgs[idx_q]-seq, db_imgs[idx_q]+seq):
            db_idx = str(idx_d).zfill(6) + ".png"
            # images_to_process[2*i] = os.path.join(db_L_dir, db_idx)
            # images_to_process[2*i+1] = os.path.join(db_R_dir, db_idx)
            images_to_process[i] = os.path.join(db_L_dir, db_idx)
            i += 1
        images_to_process.append(query_L_path[idx_q])
        images_to_process.append(query_R_path[idx_q])

        # 3-2. Select db images by location of result of step 2
        # 3-3. Select db images by graph of result of step 2

        # 4. Align photos
        print("Align photos")
        ba_api = BA_api()
        cam_L, eo_L, cam_R, eo_R = ba_api.alignphotos(images_to_process, region)

        if len(eo_L) == 0 or len(eo_R) == 0:
            f.write(query_L_idx + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t"
                    + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t"
                    + query_R_idx + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t"
                    + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0) + "\n")
        else:
            f.write(query_L_idx + "\t" + str(eo_L[0]) + "\t" + str(eo_L[1]) + "\t" + str(eo_L[2]) + "\t"
                    + str(eo_L[3]) + "\t" + str(eo_L[4]) + "\t" + str(eo_L[5]) + "\t"
                    + str(eo_L[6]) + "\t" + str(eo_L[7]) + "\t" + str(eo_L[8]) + "\t"
                    + query_R_idx + "\t" + str(eo_R[0]) + "\t" + str(eo_R[1]) + "\t" + str(eo_R[2]) + "\t"
                    + str(eo_R[3]) + "\t" + str(eo_R[4]) + "\t" + str(eo_R[5]) + "\t"
                    + str(eo_R[6]) + "\t" + str(eo_R[7]) + "\t" + str(eo_R[8]) + "\n")

        # 5. System calibration
        print("System calibration")

    f.close()
