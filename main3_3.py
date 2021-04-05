import numpy as np
import os
import glob
from ba import BA_api
import time

# train: left/right 영상을 모두 이용해 align
# test: left/right 마지막과 근처 n장의 영상을 이용, superglue 이용, matches에 test 영상 추가

start_time = time.time()

number_of_imgs = 10
regions = ["yeouido", "pangyo"]
for region in regions:
    f = open(region + "_superglue2_" + str(number_of_imgs) + ".csv", "w")
    f.write("Label_L" + "," + "X(m)" + "," + "Y(m)" + "," + "Z(m)" + ","
            + "Yaw(deg)" + "," + "Pitch(deg)" + "," + "Roll(deg)" + ","
            + "Omega(deg)" + "," + "Phi(deg)" + "," + "Kappa(deg)" + ","
            + "Label_R" + "," + "X(m)" + "," + "Y(m)" + "," + "Z(m)" + ","
            + "Yaw(deg)" + "," + "Pitch(deg)" + "," + "Roll(deg)" + ","
            + "Omega(deg)" + "," + "Phi(deg)" + "," + "Kappa(deg)" + "\n")

    # 1. Load a csv file
    reader = np.loadtxt("netvlad/retrieved naverlabs " + region + " image indices.csv", dtype=np.int, delimiter=",")

    # 2. Match the closest db images corresponding to query images
    query_imgs = reader[:, 0]
    db_imgs = reader[:, 1]

    query_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/test"
    query_L_path = ["" for i in range(len(query_imgs) * number_of_imgs)]
    query_R_path = ["" for i in range(len(query_imgs) * number_of_imgs)]
    for idx_q in range(len(query_imgs)):
        query_path = glob.glob(os.path.join(query_dir, region + str(idx_q).zfill(2)) + "/*.png")
        query_L_path[idx_q * number_of_imgs:idx_q * number_of_imgs + number_of_imgs] = query_path[-2:-2-number_of_imgs*2:-2]
        query_R_path[idx_q * number_of_imgs:idx_q * number_of_imgs + number_of_imgs] = query_path[-1:-2-number_of_imgs*2+1:-2]

        query_L_label = ["" for i in range(number_of_imgs)]
        query_R_label = ["" for i in range(number_of_imgs)]
        for idx_img in range(number_of_imgs):
            query_L_label[idx_img] = query_L_path[idx_q * number_of_imgs + idx_img]
            query_R_label[idx_img] = query_R_path[idx_q * number_of_imgs + idx_img]

        # 3. Select db images using result of step 2
        # 3-1. Select front & back 50 db images of result of step 2
        db_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/train/images/all"
        seq = 25
        i = 0
        images_to_process_train = ["" for i in range(seq * 2 * 2)]
        for idx_d in range(db_imgs[idx_q]-seq, db_imgs[idx_q]+seq):
            db_L_idx = str(idx_d).zfill(6) + "_L.png"
            db_R_idx = str(idx_d).zfill(6) + "_R.png"
            images_to_process_train[2*i] = os.path.join(db_dir, db_L_idx)
            images_to_process_train[2*i+1] = os.path.join(db_dir, db_R_idx)
            i += 1
        # images_to_process_test = []
        # for idx_img in range(number_of_imgs):
        #     images_to_process_test.append(query_L_path[idx_q * number_of_imgs + idx_img])
        #     images_to_process_test.append(query_R_path[idx_q * number_of_imgs + idx_img])

        # 3-2. Select db images by location of result of step 2
        # 3-3. Select db images by graph of result of step 2

        # 4. Align photos
        print("Align photos")
        ba_api = BA_api()
        cam_L, eo_L, cam_R, eo_R = ba_api.alignphotos_3_3(images_to_process_train, query_L_label, query_R_label,
                                                          number_of_imgs, seq, idx_q, region)

        query_L_label = query_L_label[0].split("\\")[1] + "/" + query_L_label[0].split("\\")[2]
        query_R_label = query_R_label[0].split("\\")[1] + "/" + query_R_label[0].split("\\")[2]
        if len(eo_L) == 0 or len(eo_R) == 0:    # No value
            f.write(query_L_label + "," + str(0) + "," + str(0) + "," + str(0) + ","
                    + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + ","
                    + query_R_label + "," + str(0) + "," + str(0) + "," + str(0) + ","
                    + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "\n")
        else:
            f.write(query_R_label + "," + str(eo_L[0]) + "," + str(eo_L[1]) + "," + str(eo_L[2]) + ","
                    + str(eo_L[3]) + "," + str(eo_L[4]) + "," + str(eo_L[5]) + ","
                    + str(eo_L[6]) + "," + str(eo_L[7]) + "," + str(eo_L[8]) + ","
                    + query_R_label + "," + str(eo_R[0]) + "," + str(eo_R[1]) + "," + str(eo_R[2]) + ","
                    + str(eo_R[3]) + "," + str(eo_R[4]) + "," + str(eo_R[5]) + ","
                    + str(eo_R[6]) + "," + str(eo_R[7]) + "," + str(eo_R[8]) + "\n")

        # 5. System calibration
        print("System calibration")

    f.close()

print("Elaplsed time:", time.time() - start_time)
