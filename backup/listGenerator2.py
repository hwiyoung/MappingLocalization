import numpy as np
import os
import glob
import time

# train: left/right 영상을 모두 이용해 align
# test: left/right 마지막과 근처 n장의 영상을 이용, superglue 이용

start_time = time.time()

number_of_imgs = 10
regions = ["yeouido", "pangyo"]
for region in regions:
    # 1. Load a csv file
    reader = np.loadtxt("../netvlad/retrieved naverlabs " + region + " image indices.csv", dtype=np.int, delimiter=",")

    # 2. Match the closest db images corresponding to query images
    query_imgs = reader[:, 0]
    db_imgs = reader[:, 1]

    query_dir = "Z:/Mapping_Localization/outdoor/outdoor_dataset/outdoor_main/" + region + "/test"
    for idx_q in range(len(query_imgs)):
        f = open("superglue_list/" + region + str(idx_q).zfill(2) + ".txt", "w")

        query_path = glob.glob(os.path.join(query_dir, region + str(idx_q).zfill(2)) + "/*.png")

        interval = 10
        test_img_49L = query_path[-2].split("\\")[1] + query_path[-2].split("\\")[2].zfill(9)
        test_img_49R = query_path[-2].split("\\")[1] + query_path[-1].split("\\")[2].zfill(9)
        test_img_48L = query_path[-2].split("\\")[1] + query_path[-4].split("\\")[2].zfill(9)
        test_img_47L = query_path[-2].split("\\")[1] + query_path[-6].split("\\")[2].zfill(9)
        test_img_46L = query_path[-2].split("\\")[1] + query_path[-8].split("\\")[2].zfill(9)
        test_img_45L = query_path[-2].split("\\")[1] + query_path[-10].split("\\")[2].zfill(9)
        # test_L|train|train-20|train-10|train+10|train+20|test_R|test_L-1|test_L-2|test_L-3|test_L-4
        images_to_process = [test_img_49L, str(db_imgs[idx_q]).zfill(6) + ".png",
                             str(db_imgs[idx_q] - interval*2).zfill(6) + ".png",
                             str(db_imgs[idx_q] - interval).zfill(6) + ".png",
                             str(db_imgs[idx_q] + interval).zfill(6) + ".png",
                             str(db_imgs[idx_q] + interval*2).zfill(6) + ".png",
                             test_img_49R, test_img_48L, test_img_47L, test_img_46L, test_img_45L]

        # test_L    |   train
        # test_L    |   train-20
        # test_L    |   train-10
        # test_L    |   train+10
        # test_L    |   train+20
        # test_L    |   test_R(49R)
        # test_L    |   test_L-1(48L)
        # test_L    |   test_L-2(47L)
        # test_L    |   test_L-3(46L)
        # test_L    |   test_L-4(45L)
        f.write(images_to_process[0] + " " + images_to_process[1] + "\n" +
                images_to_process[0] + " " + images_to_process[2] + "\n" +
                images_to_process[0] + " " + images_to_process[3] + "\n" +
                images_to_process[0] + " " + images_to_process[4] + "\n" +
                images_to_process[0] + " " + images_to_process[5] + "\n" +
                images_to_process[0] + " " + images_to_process[6] + "\n" +
                images_to_process[0] + " " + images_to_process[7] + "\n" +
                images_to_process[0] + " " + images_to_process[8] + "\n" +
                images_to_process[0] + " " + images_to_process[9] + "\n" +
                images_to_process[0] + " " + images_to_process[10])
        f.close()

print("Elaplsed time:", time.time() - start_time)
