"""
Generate KITTI Eigen split data with weather effects such as rain, snow, fog, speed blur

python gen_kitti_weather_data.py
"""
import PIL.Image as pil
from PIL import ImageEnhance
import os
import argparse
import numpy as np
import random
import copy
from torchvision.transforms import ToPILImage
from pba.augmentationsX import add_rain, add_snow, add_fog, add_speed

toPIL = ToPILImage()


def add_random_rain(img, seed=0):
    bright_coeff = 0.8
    img = ImageEnhance.Brightness(img.convert('RGB')).enhance(bright_coeff)
    img_data = [np.array(img)]
    # generate random slant for rain droplets
    np.random.seed(seed)
    slant = np.random.randint(-20, 21)
    img_data = add_rain(img_data, slant=slant)
    img_data = [toPIL(img).convert('RGB') for img in img_data]

    return img_data[0]


def add_random_snow(img, seed=0):
    img_data = [np.array(img)]
    np.random.seed(seed)
    snow_coeff = np.random.uniform(0, 0.51)
    img_data = add_snow(img_data, snow_coeff=snow_coeff)
    img_data = [toPIL(img).convert('RGB') for img in img_data]

    return img_data[0]


def add_random_fog(img, seed=0):
    img_data = [np.array(img)]
    np.random.seed(seed)
    _coeff = np.random.uniform(0, 0.41)
    fog_coeff = 0.2 + _coeff
    img_data = add_fog(img_data, fog_coeff=fog_coeff)
    img_data = [toPIL(img).convert('RGB') for img in img_data]

    return img_data[0]


def add_speed_blur(img, seed=0):
    img_data = [np.array(img)]
    img_data = add_speed(img_data, speed_coeff=0.1)
    img_data = [toPIL(img).convert('RGB') for img in img_data]

    return img_data[0]


def read_test_files(dataset_dir, test_file_path):
    """
    Read kitti eigen test files
    :param dataset_dir: root path of kitti dataset
    :return: read test files with kitti root path added as prefix
    """
    with open(test_file_path, 'r') as f:
        test_files = f.readlines()
        test_files = [t.rstrip() for t in test_files]
        test_files = [os.path.join(dataset_dir, t) for t in test_files]

    return test_files


def main(args):
    augmentations = [add_random_rain, add_random_snow, add_random_fog, add_speed_blur]
    test_files = read_test_files(args.kitti_raw, args.test_file_path)
    # test_files = test_files[:5]

    # process files
    for idx, fname in enumerate(test_files):
        print("processing: {}/{}".format(idx+1, len(test_files)))
        fh = open(fname, 'rb')
        img = pil.open(fh)
        apply_x = copy.copy(augmentations)
        random.shuffle(apply_x)
        # count = np.random.choice([0, 1, 2], p=[0.10, 0.45, 0.45])
        count = np.random.choice([0, 1], p=[0.10, 0.90])
        if count != 0:
            for aug_fun in apply_x:
                img = aug_fun(img, seed=idx)
                count -= 1
                if count == 0:
                    break
        else:
            pass  # no augmentation applied

        # write augmented sample on disk
        new_fname = fname.replace(args.kitti_raw, args.save_dir)
        if not os.path.exists(os.path.dirname(new_fname)):
            os.makedirs(os.path.dirname(new_fname))

        # print(new_fname)
        img.save(new_fname)

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Dataset related flags
    parser.add_argument(
        '--kitti_raw',
        default='/ceph/amanraj/data/kitti_raw_eigen_test',
        help='directory where raw KITTI dataset is located.'
    )
    parser.add_argument(
        '--test_file_path',
        default='/ceph/amanraj/data/kitti_raw_eigen_test/test_files_eigen.txt',
        help='.txt file containing list of kitti eigen test files'
    )
    parser.add_argument(
        '--save_dir',
        default='/ceph/amanraj/data/kitti_raw_eigen_test_weather_single',
        help='directory to save generated dataset'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

