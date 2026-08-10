import numpy as np
import torch
from torch.utils.data import Dataset
from feeder import tools
import sys
import random

### ver2 multitest ###
class Feeder_multitest(Dataset): 
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, source_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 frame_noise=[0., 0.], joint_noise=[0., 0.], impulse_noise=0., motion_noise=0., flip=False,
                 test_num_samples=10, ###
                 bone=False, vel=False):
        """
        source: data_numpy_aug
        target: data_numpy
        source & target share same crop, flip, rot
        the only difference is that source add synthetic noise

        frame_noise / joint_noise: (p, std)
        impulse_noise: p
        motion_noise: std
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose # not used
        self.random_shift = random_shift # not used
        self.random_move = random_move # not used
        self.frame_noise = frame_noise
        self.joint_noise = joint_noise
        self.impulse_noise = impulse_noise
        self.motion_noise = motion_noise
        self.flip = flip
        self.window_size = window_size
        self.normalization = normalization
        assert not self.normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.source_rot = source_rot

        self.bone = bone
        self.vel = vel
        self.load_data()
        self.test_num_samples = test_num_samples
        
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


    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        
        if self.split == 'train': # same as Feeder
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
            
            if self.flip:
                data_numpy = tools.random_spatial_flip(data_numpy, p=0.5, dataset='ntu')

            if self.random_rot:
                data_numpy = tools.random_rot(data_numpy)
            
            # generate noisy data
            data_numpy_aug = self.add_source_aug(data_numpy)
        
            return data_numpy, data_numpy_aug, label, index  

        else: # multiple samples for testing
            data_list, data_list_aug = [], []

            for s in range(self.test_num_samples):
                d = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
                
                if self.flip:
                    d = tools.random_spatial_flip(d, p=0.5, dataset='ntu')

                if self.random_rot:
                    d = tools.random_rot(d)
                
                # generate noisy data
                d_aug = self.add_source_aug(d)

                data_list.append(d)
                data_list_aug.append(d_aug)
            
            return data_list, data_list_aug, label, index  
            

    def add_source_aug(self, data_numpy):
        
        # rot
        if self.source_rot:
            data_numpy_aug = tools.random_rot(data_numpy)
        else:
            data_numpy_aug = data_numpy.copy()
        
        # noise
        if self.motion_noise > 0.:   
            data_numpy_aug = tools.motion_noise(data_numpy_aug, std=self.motion_noise)
        if self.frame_noise[0] > 0. and self.frame_noise[1] > 0.: 
            data_numpy_aug = tools.frame_noise(data_numpy_aug, p=self.frame_noise[0], std=self.frame_noise[1], num_frames=self.window_size)
        if self.joint_noise[0] > 0. and self.joint_noise[1] > 0.:    
            data_numpy_aug = tools.joint_noise(data_numpy_aug, p=self.joint_noise[0], std=self.joint_noise[1], num_joints=25)
        if self.impulse_noise > 0.:  
            data_numpy_aug = tools.impulse_noise(data_numpy_aug, p=self.impulse_noise)
        
        return data_numpy_aug


### ver2 ###
class Feeder(Dataset): 
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, source_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 frame_noise=[0., 0.], joint_noise=[0., 0.], impulse_noise=0., motion_noise=0., flip=False,
                 bone=False, vel=False):
        """
        source: data_numpy_aug
        target: data_numpy
        source & target share same crop, flip, rot
        the only difference is that source add synthetic noise

        frame_noise / joint_noise: (p, std)
        impulse_noise: p
        motion_noise: std
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose # not used
        self.random_shift = random_shift # not used
        self.random_move = random_move # not used
        self.frame_noise = frame_noise
        self.joint_noise = joint_noise
        self.impulse_noise = impulse_noise
        self.motion_noise = motion_noise
        self.flip = flip
        self.window_size = window_size
        self.normalization = normalization
        assert not self.normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.source_rot = source_rot

        self.bone = bone
        self.vel = vel
        self.load_data()

        self.expanding_dataset = False
        self.new_data = []
        self.new_label = []
        
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
        print('Dataset number of classes', self.label.max()+1)


    def __len__(self):
        return len(self.label) + len(self.new_label)

    def __iter__(self):
        return self

    def _load_processed(self, index):
        if index < len(self.label):
            data_numpy = self.data[index]
            label = self.label[index]
        else:
            data_numpy = self.new_data[index-len(self.label)]
            label = self.new_label[index-len(self.label)]           
        data_numpy = np.array(data_numpy)
        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        p_interval = self.p_interval if not self.expanding_dataset else [0.95]
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, p_interval, self.window_size)
        
        if self.flip and not self.expanding_dataset:
            data_numpy = tools.random_spatial_flip(data_numpy, p=0.5, dataset='ntu')

        if self.random_rot and not self.expanding_dataset:
            data_numpy = tools.random_rot(data_numpy)
        return data_numpy, label

    def __getitem__(self, index):
        data_numpy, label = self._load_processed(index)
        # generate noisy data
        data_numpy_aug = self.add_source_aug(data_numpy)
        return data_numpy, data_numpy_aug, label, index

    def add_source_aug(self, data_numpy):
        
        # rot
        if self.source_rot:
            data_numpy_aug = tools.random_rot(data_numpy)
        else:
            data_numpy_aug = data_numpy.copy()
        
        # noise
        if self.motion_noise > 0.:   
            data_numpy_aug = tools.motion_noise(data_numpy_aug, std=self.motion_noise)
        if self.frame_noise[0] > 0. and self.frame_noise[1] > 0.: 
            data_numpy_aug = tools.frame_noise(data_numpy_aug, p=self.frame_noise[0], std=self.frame_noise[1], num_frames=self.window_size)
        if self.joint_noise[0] > 0. and self.joint_noise[1] > 0.:    
            data_numpy_aug = tools.joint_noise(data_numpy_aug, p=self.joint_noise[0], std=self.joint_noise[1], num_joints=25)
        if self.impulse_noise > 0.:  
            data_numpy_aug = tools.impulse_noise(data_numpy_aug, p=self.impulse_noise)
        
        return data_numpy_aug

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

        for _ in range(10):
            indices = np.arange(len(self.label))
            np.random.shuffle(indices)
            
            for idx in indices:
                
                if len(new_data) > args.generated_ratio * len(self.label):
                    break
                
                data_numpy, _, label, _ = self.__getitem__(idx)
                if (idx+1) % 200 == 0:
                    print(f'Generated {idx+1}/{len(self.label)} samples')
                    sys.stdout.flush()
                data = torch.from_numpy(data_numpy).to(device).unsqueeze(0)
                with torch.no_grad():
                    data_mod = model.random_modify(
                        data, 
                        use_z=args.use_z, 
                        z_noise_std=args.z_noise_std, 
                        t_start=args.t_start,
                        sample=args.sample,
                    )
                    _, C, T, V, M = data_mod.shape
                    data_mod = torch.cat([data_mod, torch.zeros(1, C, self.data.shape[2]-T, V, M, device=data_mod.device)], dim=2)
                data_mod = data_mod.cpu().numpy()
                new_data.append(data_mod)
                new_label.append(label)
        print('stage1')
        sys.stdout.flush()
        new_data = np.concatenate(new_data, axis=0) #NCTVM
        print('stage2')
        sys.stdout.flush()
        new_label = np.array(new_label)
        print(new_data.shape)
        sys.stdout.flush()
        N, C, T, V, M = new_data.shape
        #new_data = np.concatenate([new_data, np.zeros((N, C, self.data.shape[2]-T, V, M))], axis=2)
        print('stage3')
        sys.stdout.flush()
        #self.data = np.concatenate([self.data, new_data], axis=0)
        #self.label = np.concatenate([self.label, new_label], axis=0)
        self.new_data = new_data
        self.new_label = new_label

        self.expanding_dataset = False
        print(f'Finished expanding dataset. Expanded data: {len(self.label)}')
        print('#'*100)


class FeederOSE(Feeder):
    """Return a source and an epoch-frozen OSE peer without recursive loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split != 'train':
            raise ValueError('FeederOSE only supports the training split')
        self.neighbor_map = {}
        self.sample_indices = np.arange(len(self.label), dtype=np.int64)

    def exclude_ose_exemplars(self, exemplar_ids):
        excluded = {int(dataset_id) for dataset_id in exemplar_ids}
        if any(dataset_id < 0 or dataset_id >= len(self.label)
               for dataset_id in excluded):
            raise ValueError('OSE exemplar index is outside the training split')
        self.sample_indices = np.asarray([
            dataset_id for dataset_id in range(len(self.label))
            if dataset_id not in excluded
        ], dtype=np.int64)
        if self.sample_indices.size == 0:
            raise ValueError('No unlabeled samples remain after excluding exemplars')

    def __len__(self):
        return len(self.sample_indices)

    def set_neighbor_map(self, neighbor_map):
        self.neighbor_map = {
            int(source_id): [int(peer_id) for peer_id in peer_ids]
            for source_id, peer_ids in neighbor_map.items()
        }

    def get_ose_samples(self, dataset_ids):
        samples = [self._load_processed(int(dataset_id))[0] for dataset_id in dataset_ids]
        return np.stack(samples)

    def __getitem__(self, index):
        source_id = int(self.sample_indices[index])
        source, source_label = self._load_processed(source_id)
        source_aug = self.add_source_aug(source)
        candidates = self.neighbor_map.get(source_id, [])
        if candidates:
            peer_id = random.choice(candidates)
            peer, peer_label = self._load_processed(peer_id)
            has_peer = True
        else:
            peer_id = source_id
            peer = source.copy()
            peer_label = source_label
            has_peer = False
        # Labels are returned only for offline routing diagnostics in the
        # training engine. They are never passed to the model or a loss.
        return (
            source, source_aug, peer, source_id, peer_id, has_peer,
            source_label, peer_label,
        )

### MAMP code ###
class Feeder_original(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False):
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
        if normalization:
            self.get_mean_map()

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


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        print(f'Data Normalization is on. Minimal std: {np.min(self.std_map)}')
        self.std_map[self.std_map < 0.1] = 0.1

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_rot:
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

        return data_numpy, label, index
