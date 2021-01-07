import os
from os.path import join

import numpy as np
from PIL import Image
import random


def read_data(path_, suffix='.png'):
    image_list = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(path_):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))

        for file in files:
            if file.endswith(suffix):
                print(len(path) * '---', file)
                image_list.append(np.asarray(Image.open(path_ + '/' + os.path.basename(root) + '/' + file)))

    return image_list


def pad_image(img_3d, img_2d):
    height, width = img_2d.shape

    padded_img_3d = np.zeros((height + 80, width + 80, 3), dtype='uint8')
    padded_image_2d = np.zeros((height + 80, width + 80), dtype='uint8')
    padded_img_3d[40:height + 40, 40:width + 40, :] = img_3d
    padded_image_2d[40:height + 40, 40:width + 40] = img_2d
    return padded_image_2d, padded_img_3d


def crope(img_3d, img_2d):
    padded_image_2d, padded_img_3d = pad_image(img_3d, img_2d)

    indexes = np.argwhere(19 == padded_image_2d)

    if 0 == indexes.size:
        return

    tfl_x, tfl_y = random.choice(indexes)
    tfl = padded_img_3d[tfl_x - 40:tfl_x + 41, tfl_y - 40:tfl_y + 41, :]

    check_not_tfl = padded_image_2d[tfl_x - 40:tfl_x + 41, tfl_y - 40:tfl_y + 41]

    while check_not_tfl[19 == check_not_tfl].size > 0:
        not_tfl_x, not_tfl_y = random.choice(np.argwhere(19 != padded_image_2d[40:-40, 40:-40])) + 40
        not_tfl = padded_img_3d[not_tfl_x - 40:not_tfl_x + 41, not_tfl_y - 40:not_tfl_y + 41]
        check_not_tfl = padded_image_2d[not_tfl_x - 40:not_tfl_x + 41, not_tfl_y - 40:not_tfl_y + 41]
    return [tfl, not_tfl]


def prepare_data_set(sample_path, labeled_path):
    sample_imgs = read_data(sample_path, '_leftImg8bit.png')
    labeled_imgs = read_data(labeled_path, '_gtFine_labelIds.png')

    data_set = []

    for sample, labeled in zip(sample_imgs, labeled_imgs):
        data_set.append(crope(sample, labeled))

    data_set = sum([x for x in data_set if x], [])

    return data_set


def write_to_bin_file(images, is_tfl, dirc):
    np.array(images, dtype='uint8').tofile('./data/' + dirc + '/data.bin')
    np.array(is_tfl, dtype='uint8').tofile('./data/' + dirc + '/labels.bin')


data = prepare_data_set('data/CityScapes/leftImg8bit/train', 'data/CityScapes/gtFine/train')
write_to_bin_file(data, np.array([i % 2 == 0 for i in range(len(data))]), 'train')

data = prepare_data_set('data/CityScapes/leftImg8bit/val', 'data/CityScapes/gtFine/val')
write_to_bin_file(data, np.array([i % 2 == 0 for i in range(len(data))]), 'val')


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}

data_dir = 'data'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}

for k, v in datasets.items():
    print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))
