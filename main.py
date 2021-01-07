# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from operator import itemgetter

import numpy as np

import SFM
from SFM_standAlone import FrameContainer, visualize
from file_utils import read_txt, read_pickle
from part2 import crop_imgs
from run_attention import find_tfl_lights

from matplotlib.keras.models import load_model


class Data:
    def __init__(self):
        self.prev = None
        self.curr = None
        self.focal = None
        self.pp = None

    def init_f_pp(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def update(self, curr):
        self.prev = self.curr
        self.curr = curr


class ProcessData:
    def __init__(self, pls_file):
        self.pls_file = pls_file

    def process_data(self, frames):
        lines = read_txt(self.pls_file)
        pkl_path = lines[0]
        data = read_pickle(pkl_path)

        frames.init_f_pp(data['flx'], data['principle_point'])

        for img in lines[1:]:
            curr_frame_id = int(img[31:33])
            curr_img_path = img
            curr_container = FrameContainer(curr_img_path)
            curr_container.id = curr_frame_id
            # curr_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])
            EM = np.eye(4)

            for i in range(curr_frame_id - 1, curr_frame_id):
                EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            curr_container.EM = EM
            frames.update(curr_container)
            yield

    def count_lines(self):
        lines = read_txt(self.pls_file)
        return len(lines) - 1

    def first_id(self):
        lines = read_txt(self.pls_file)
        return int(lines[1][31:33])


class Adapter1:
    def filter_same_candidates(self, all_candidates):
        red_x, red_y, green_x, green_y = all_candidates[0], all_candidates[1], all_candidates[2], all_candidates[3]
        for r_x, r_y in zip(red_x, red_y):
            for g_x, g_y in zip(green_x, green_y):
                if r_x == g_x and r_y == g_y:
                    green_x -= g_x
                    green_y -= g_y

        return red_x, red_y, green_x, green_y

    def adapt(self, all_candidates):
        red_x, red_y, green_x, green_y = self.filter_same_candidates(all_candidates)
        candidates = []
        auxiliary = []
        for x, y in zip(red_x, red_y):
            candidates.append([x, y])
            auxiliary.append("r")
        for x, y in zip(green_x, green_y):
            candidates.append([x, y])
            auxiliary.append("g")

        return np.array(candidates), auxiliary


class TFLManager:
    def run(self, data):
        adapter1 = Adapter1()
        image = np.array(data.curr.img)
        candidates, auxiliary = adapter1.adapt(find_tfl_lights(image))

        cropped_imgs = crop_imgs(image, candidates)
        # candidates = [i for i in candidates if i not in not_imgs_ind]# update the candidates according to the cropped_imgs

        loaded_model = load_model("model.h5")
        percents = loaded_model.predict(np.array(cropped_imgs))

        tmp = []
        for i, percent in enumerate(percents[:, 1]):
            if percent > 0.8:
                tmp.append(candidates[i])
                # tmp.append([candidates[i], auxiliary[i], percent])

        # sorted(tmp, key=lambda k: k[2], reverse=True)
        tfl_candidates = np.array(tmp[:20])
        # tfl_auxiliary = np.array(tmp)[:20, 1]
        data.curr.traffic_light = tfl_candidates

        if data.prev and percents.shape[0] <= len(cropped_imgs):
            curr = SFM.calc_TFL_dist(data.prev, data.curr, data.focal, data.pp)
            visualize(data.prev, curr, data.focal, data.pp)


class Controller:
    def process(self, pls_path):
        data = Data()
        p = ProcessData(pls_path)
        manager = TFLManager()

        for _ in p.process_data(data):
            manager.run(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    my_controller = Controller()
    my_controller.process("data/pls_imgs/pls.txt")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# todo: if cropped imgs are allmost same need to filter them
# todo: sort all imgs according to percents then take the 10 largest
