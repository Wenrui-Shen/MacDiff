import numpy as np
import torch
from torch.utils.data import Dataset
from feeder import tools
import time
import copy

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_ratio=None,
                 random_choose=False, random_shift=False, random_move=False, random_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, bone=False, vel=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        
        self.load_data()

        if data_ratio is not None:
            self.random_select_data(data_ratio)

        if normalization:
            self.get_mean_map()

        self.expanding_dataset = False


    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.original_len = len(self.data) ###

    def random_select_data(self, data_ratio):
        N = self.data.shape[0]
        idx = np.arange(N)

        #seed_value = int(time.time())
        #np.random.seed(seed_value)
        np.random.shuffle(idx)

        N_used = int(N * data_ratio)
        idx_used = idx[ :N_used]
        print('total samples:', N, 'use samples:',N_used)
        print('use idx', idx_used)
        #idx_used = idx[-N_used:]
        
        self.data = self.data[idx_used]
        self.label = self.label[idx_used]
        # self.sample_name = self.sample_name[idx_used]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index, ):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        # p_interval = self.p_interval if not self.expanding_dataset else [0.95]
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot: # and not self.expand_dataset:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        #return data_numpy, label, index
        return data_numpy, data_numpy.copy(), label, index


    def precalculate_latent(self, model, device):
        self.latent = []
        for idx in range(len(self.label)):
            data_numpy, _, label, _ = self.__getitem__(idx)
            data = torch.from_numpy(data_numpy).to(device).unsqueeze(0)
            with torch.no_grad():
                z = model.get_latent(data).cpu().numpy()
                self.latent.append(z)
        self.latent = np.concatenate(self.latent, axis=0) # [N, M, C]


    def expand_dataset(self, model, device, args):
        print('#'*100)
        print(f'Expanding dataset. Expand ratio: {args.generated_ratio}. Initial data: {len(self.label)}')
        self.expanding_dataset = True
        new_data, new_label = [], []

        for _ in range(args.generated_ratio):
            for idx in range(len(self.label)):
                data_numpy, _, label, _ = self.__getitem__(idx)
                data = torch.from_numpy(data_numpy).to(device).unsqueeze(0)
                with torch.no_grad():
                    data_mod = model.random_modify(
                        data, 
                        use_z=args.use_z, 
                        z_noise_std=args.z_noise_std, 
                        t_start=args.t_start,
                        sample=args.sample,
                    )
                data_mod = data_mod.cpu().numpy()
                new_data.append(data_mod)
                new_label.append(label)
        
        new_data = np.concatenate(new_data, axis=0) #NCTVM
        new_label = np.array(new_label)
        N, C, T, V, M = new_data.shape
        new_data = np.concatenate([new_data, np.zeros((N, C, self.data.shape[2]-T, V, M))], axis=2)
        self.data = np.concatenate([self.data, new_data], axis=0)
        self.label = np.concatenate([self.label, new_label], axis=0)
        self.expanding_dataset = False
        print(f'Finished expanding dataset. Expanded data: {len(self.label)}')
        print('#'*100)

