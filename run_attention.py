try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    import cv2

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def find_lights_indexes(image):
    indices = np.where(np.all(image != 0, axis=-1))
    return indices[1], indices[0]


def find_green_lights(image):
    result = image.copy()
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 130, 100])
    upper = np.array([255, 179, 255])
    mask = cv2.inRange(new_image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    return result


def find_red_lights(image):
    result = image.copy()
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([120, 25, 0])
    upper = np.array([179, 255, 100])
    mask = cv2.inRange(new_image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    return result


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    res = find_red_lights(c_image)
    x_red, y_red = find_lights_indexes(res)

    res = find_green_lights(c_image)
    x_green, y_green = find_lights_indexes(res)

    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
