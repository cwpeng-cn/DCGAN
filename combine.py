import os
import shutil

ALL_PATH = "../datasets/selfie2anime/all",
TRAINB_PATH = "../datasets/selfie2anime/trainB"
TESTB_PATH = "../datasets/selfie2anime/testB"


def rename_copy(src_folder, des_folder):
    for name in os.listdir(src_folder):
        suffix = "." + name.split(".")[-1]
        src_path = os.path.join(src_folder, name)
        des_path = os.path.join(des_folder, str(len(os.listdir(des_folder))) + suffix)
        shutil.copy(src_path, des_path)


if not os.path.exists(ALL_PATH):
    os.mkdir(ALL_PATH)
rename_copy(TESTB_PATH, ALL_PATH)
rename_copy(TRAINB_PATH, ALL_PATH)
