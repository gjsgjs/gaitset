import torch.utils.data as tordata
import random


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            # import pdb
            # pdb.set_trace()
            sample_indices = list()
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0])
            # 从100个人中取样batch_size[0](4)个不同的人,
            for pid in pid_list:
                # person对应的视频(假64真16)
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                # 从一个person取self.batch_size[1](4)个不同的视频
                sample_indices += _index
                # 里面存抽取的序列
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
