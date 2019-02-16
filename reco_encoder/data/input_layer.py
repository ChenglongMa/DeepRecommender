# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from os import listdir, path
from random import shuffle

import torch


class UserItemRecDataProvider:
    def __init__(self, params, user_id_map=None, item_id_map=None):
        self._params = params
        self._data_dir = self.params['data_dir']
        self._extension = ".txt" if 'extension' not in self.params else self.params['extension']
        self._i_id = 0 if 'itemIdInd' not in self.params else self.params['itemIdInd']
        self._u_id = 1 if 'userIdInd' not in self.params else self.params['userIdInd']
        self._r_id = 2 if 'ratingInd' not in self.params else self.params['ratingInd']
        self._major = 'items' if 'major' not in self.params else self.params['major']
        self._header = 0 if 'header' not in self.params else self.params['header']
        if not (self._major == 'items' or self._major == 'users'):
            raise ValueError("Major must be 'users' or 'items', but got {}".format(self._major))

        self._major_ind = self._i_id if self._major == 'items' else self._u_id
        self._minor_ind = self._u_id if self._major == 'items' else self._i_id
        self._delimiter = '\t' if 'delimiter' not in self.params else self.params['delimiter']

        if user_id_map is None or item_id_map is None:
            self._build_maps()
        else:
            self._user_id_map = user_id_map
            self._item_id_map = item_id_map

        major_map = self._item_id_map if self._major == 'items' else self._user_id_map
        minor_map = self._user_id_map if self._major == 'items' else self._item_id_map
        self._vector_dim = len(minor_map)

        # TODO: change here
        if path.isfile(self._data_dir):
            src_files = [self._data_dir]
        else:
            src_files = [path.join(self._data_dir, f)
                         for f in listdir(self._data_dir)
                         if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

        self._batch_size = self.params['batch_size']

        self.data = dict()

        for source_file in src_files:
            with open(source_file, 'r') as src:
                for line in src.readlines():
                    for i in range(self._header):
                        continue
                    parts = line.strip().split(self._delimiter)
                    if len(parts) < 3:
                        raise ValueError('Encountered badly formatted line in {}'.format(source_file))
                    key = major_map[int(parts[self._major_ind])]
                    value = minor_map[int(parts[self._minor_ind])]
                    rating = float(parts[self._r_id])
                    # print("Key: {}, Value: {}, Rating: {}".format(key, value, rating))
                    if key not in self.data:
                        self.data[key] = []
                    self.data[key].append((value, rating))

    def _build_maps(self):
        self._user_id_map = dict()
        self._item_id_map = dict()

        if path.isfile(self._data_dir):
            src_files = [self._data_dir]
        else:
            src_files = [path.join(self._data_dir, f)
                         for f in listdir(self._data_dir)
                         if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

        u_id = 0
        i_id = 0
        for source_file in src_files:
            with open(source_file, 'r') as src:
                for line in src.readlines():
                    for i in range(self._header):
                        continue
                    parts = line.strip().split(self._delimiter)
                    if len(parts) < 3:
                        raise ValueError('Encountered badly formatted line in {}'.format(source_file))

                    u_id_orig = int(parts[self._u_id])
                    if u_id_orig not in self._user_id_map:
                        self._user_id_map[u_id_orig] = u_id
                        u_id += 1

                    i_id_orig = int(parts[self._i_id])
                    if i_id_orig not in self._item_id_map:
                        self._item_id_map[i_id_orig] = i_id
                        i_id += 1

    def iterate_one_epoch(self):
        data = self.data
        keys = list(data.keys())
        shuffle(keys)
        start_idx = 0
        end_idx = self._batch_size
        # Convert dict to dense matrix
        # columns: user_id, item_id, rating
        while end_idx < len(keys):
            local_ind = 0  # user id start with 0
            user_idx = []
            item_idx = []  # NOTE: take item-based for example
            ratings = []
            for ind in range(start_idx, end_idx):
                item_idx += [v[0] for v in data[keys[ind]]]
                user_idx += [local_ind] * len([v[0] for v in data[keys[ind]]])
                ratings += [v[1] for v in data[keys[ind]]]
                local_ind += 1

            i_torch = torch.LongTensor([user_idx, item_idx])
            v_torch = torch.FloatTensor(ratings)

            mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([self._batch_size, self._vector_dim]))
            start_idx += self._batch_size
            end_idx += self._batch_size
            yield mini_batch

    def iterate_one_epoch_eval(self, for_inf=False):
        keys = list(self.data.keys())
        idx = 0
        while idx < len(keys):
            # TODO: why here is [0]
            user_idx = [0] * len([v[0] for v in self.data[keys[idx]]])
            item_idx = [v[0] for v in self.data[keys[idx]]]
            ratings = [v[1] for v in self.data[keys[idx]]]

            src_user_idx = [0] * len([v[0] for v in self.src_data[keys[idx]]])
            src_item_idx = [v[0] for v in self.src_data[keys[idx]]]
            src_ratings = [v[1] for v in self.src_data[keys[idx]]]

            i_torch = torch.LongTensor([user_idx, item_idx])
            v_torch = torch.FloatTensor(ratings)

            src_i_torch = torch.LongTensor([src_user_idx, src_item_idx])
            src_v_torch = torch.FloatTensor(src_ratings)

            mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([1, self._vector_dim])),
                          torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([1, self._vector_dim])))
            idx += 1
            if not for_inf:
                yield mini_batch
            else:
                yield mini_batch, keys[idx - 1]

    @property
    def vector_dim(self):
        return self._vector_dim

    @property
    def userIdMap(self):
        return self._user_id_map

    @property
    def itemIdMap(self):
        return self._item_id_map

    @property
    def params(self):
        return self._params
