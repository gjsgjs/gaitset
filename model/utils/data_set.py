import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import cv2
import xarray as xr
from tqdm import tqdm
import pickle


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)  # 去重 ，保存所有的人物标签
        self.seq_type_set = set(self.seq_type)  # 去重，保存最终的种类（bg-01。。。）
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in tqdm(range(self.data_size)):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index]

    def __len__(self):
        return len(self.label)


class DataSet2(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)  # 100
        self.seq_type_set = set(self.seq_type)  # 64
        self.view_set = set(self.view)  # '000' 1
        # (100, 64, 1)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        # (100, 64, 1) 索引为coords，找到为对应下标i，找不到为-1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in tqdm(range(self.data_size)):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    # def __loader__(self, path):
    #     return self.img2xarray(
    #         path)[:, :, self.cut_padding:-self.cut_padding].astype(
    #         'float32') / 255.0

    def __loader__(self, path):
        return self.pkl2array(path)[:, :, self.cut_padding:-self.cut_padding].astype('float32') / 255.0

    def pkl2array(self, file_path):
        pkls = sorted(list(os.listdir(file_path)))
        pkl = pkls[0]
        frame_list = []
        if osp.isfile(osp.join(file_path, pkl)):
            with open(osp.join(file_path, pkl), 'rb') as f:
                data = pickle.load(f)
                for i in range(data.shape[0]):
                    frame_list.append(data[i, :, :])

        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __getitem__(self, index):
        # pose sequence sampling
        #  batch_sampler=triplet_sampler采样的index
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            # 训练开始时调用 def load_all_data(self): 所有数据加载到self.data中
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:  # 训练时，直接取里面的数据
            data = self.data[index]
            frame_set = self.frame_set[index]

        # 这些数据打包进入my_collate_fn(batch)
        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index]

    def __len__(self):
        return len(self.label)
