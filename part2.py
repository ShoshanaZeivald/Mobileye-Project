import numpy as np


def pad_image(img):
    height, width, _ = img.shape
    padded_img = np.zeros((height + 81, width + 81, 3), dtype='uint8')
    padded_img[40:height + 40, 40:width + 40, :] = img
    return padded_img


def crop(img, y, x):
    padded_img = pad_image(img)
    cropped_img = padded_img[x:x + 81, y:y + 81, :]
    return cropped_img


def crop_imgs(img, cand_list):
    cropped = [crop(img, i[0], i[1]) for i in cand_list]
    return [x for x in cropped if x.shape == (81, 81, 3)]
    #cropped = [crop(img, i[0], i[1]) for i in cand_list]
    #imgs = []
    #not_imgs_ind = []
    #for i,x in enumerate(cropped):
    #    if x.shape == (81, 81, 3):
    #        imgs.append(x)
    #    else:
     #     not_imgs_ind.append(i)
    #return imgs, not_imgs_ind

