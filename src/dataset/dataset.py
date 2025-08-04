import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import json
from pathlib import Path
import random

from src.utils.helpers import (
    get_file_list, get_filename_wo_ext, get_file_list_with_extension, 
    get_or_create_file_list_json,
)
from src.utils.geometry import normalize_curves
from src.dataset.dataset_fn import (
    scale_and_jitter_pc, scale_and_jitter_wireframe_set, curve_yz_scale,
    random_viewpoint, hidden_point_removal,
    aug_pc_by_idx, gaussian_smooth_curve, compute_diffs,
)

class LatentDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = None,
        is_train: bool = True,
        replication: int = 1,
        sample: bool = False,
        condition_on_points: bool = False,
        transform=scale_and_jitter_pc,
        use_partial_pc: bool = False,
        use_sampled_zs: bool = False,
        use_pc_noise: bool = False,
        condition_on_img: bool = False,
        k=48,
        pc_dir = '/root/pc_for_condition',
        img_dir = '/root/img_latents',
        point_num = 1024,
    ):
        super().__init__()
        self.is_train = is_train
        self.replica = replication
        self.data_path = dataset_file_path
        self.sample = sample
        self.condition_on_points = condition_on_points
        self.transform = transform
        self.use_partial_pc = use_partial_pc
        self.use_sampled_zs = use_sampled_zs
        self.use_pc_noise = use_pc_noise
        self.condition_on_img = condition_on_img
        self.pc_dir = pc_dir
        self.img_dir = img_dir
        self.point_num = point_num

        eval_mode = not is_train
        if self.sample or eval_mode:
            self.transform = None

        self.data, self.uids = self._load_data(dataset_file_path)
        
        print(f'load {len(self.data)} data')
        
        self.samples_per_file = k
        
        if self.condition_on_img:
            self.total_samples = len(self.data) * self.samples_per_file
        else:
            self.total_samples = len(self.data) 
        
        

    def _load_data(self, data_path):
        dataset = np.load(data_path, allow_pickle=True)
        data = dataset['data']
        uids = dataset['uids']
        return data, uids

    def __len__(self):
        if self.is_train != True:
            return self.total_samples
        else:
            return self.total_samples * self.replica

    def __getitem__(self, idx):
        idx = idx % len(self.data)

        if self.condition_on_points:            
            
            wireframe_zs_file_path = self.data[idx]
            uid = get_filename_wo_ext(wireframe_zs_file_path)
            
            if self.is_train:
                uuid = uid.split("_")[0]
                aug_idx = uid.split("_")[1]
            else:
                uuid = uid.split("_")[0]
                aug_idx = 0
                
            pc_file_path = self.pc_dir + '/' + uuid + '.npy'
            pc = np.load(pc_file_path)

            if self.use_partial_pc:
                pc = pc[np.random.choice(len(pc), 2 * self.point_num, replace=False)]

                campos = random_viewpoint()
                points = pc[:, :3]  
                indices = hidden_point_removal(points, campos, only_return_indices=True)
                pc = pc[indices]
                            
            if len(pc) < self.point_num:
                pc = pc[np.random.choice(len(pc), self.point_num, replace=True)]
            else:
                pc = pc[np.random.choice(len(pc), self.point_num, replace=False)]

            pc = aug_pc_by_idx(pc, int(aug_idx))

            if self.transform is not None:
                if self.use_pc_noise:
                    noise_level = 0.02
                else:   
                    noise_level = 0.01
                
                pc = self.transform(pc, is_rotation=True, noise_level=noise_level)
            
            pc = torch.from_numpy(pc).to(torch.float32)
            context = pc
        elif self.condition_on_img:
            file_idx = idx // self.samples_per_file
            sample_idx = idx % self.samples_per_file

            wireframe_zs_file_path = self.data[file_idx]
            uid = get_filename_wo_ext(wireframe_zs_file_path)
            uuid = uid.split("_")[0]

            img_latent_file_path = self.img_dir + '/' + uuid + '/img_feature_dinov2_' + str(sample_idx) + '.npy'
            img_latent = np.load(img_latent_file_path)

            img_latent = torch.from_numpy(img_latent).to(torch.float32)
            context = img_latent
        else:
            wireframe_zs = self.data[idx]
            context = None

        mu = wireframe_zs[:,:16]
        std = wireframe_zs[:,16:]
        zs = mu + std * np.random.randn(*std.shape)
            
        zs = torch.from_numpy(zs).to(torch.float32)

        payload = dict(zs=zs)
        
        if context is not None:
            payload['context'] = context
    
        if self.sample:
            uid = self.uids[idx]
            payload['uid'] = uid

        return payload


class WireframeDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = None,
        transform = scale_and_jitter_wireframe_set,
        is_train: bool = True,
        replication: float = 1.,
        sample: bool = False,
        max_num_lines: int = 128,
        is_curve_latent: bool = True,
    ):
        super().__init__()
        self.transform = transform
        self.is_train = is_train
        self.replica = replication
        self.sample = sample
        self.max_num_lines = max_num_lines
        self.is_curve_latent = is_curve_latent
        
        if self.sample:
            self.transform = None
        
        self.data = self._load_data(dataset_file_path)
        print(f'load {len(self.data)} valid data')
        
    def _load_data(self, dataset_path):
        with open(dataset_path, 'r') as f:
            file_path_list = json.load(f)
        
        return file_path_list

    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return int(len(self.data) * self.replica) 

    def __getitem__(self, idx):
        
        idx = idx % len(self.data)
        sample_path = self.data[idx]

        sample = np.load(sample_path)
        adjs = sample['adjs']
        vertices = sample['vertices']
        
        num_lines = adjs.shape[0]
        diffs = compute_diffs(adjs)
        segments = vertices[adjs]

        if self.is_curve_latent:
            feature = rearrange(sample['zs'], 'n h (block w) -> n (block h w)', block=2, w=3)
        else:
            norm_curves = normalize_curves(sample['edge_points'])
            feature = rearrange(norm_curves, 'b n c -> b (n c)')


        if self.transform is not None:
            segments = self.transform(segments)        
        
        segments = rearrange(segments, 'n v c -> n (v c)')
        xs = np.concatenate([segments, feature], axis=1)

        padding_cols = self.max_num_lines - num_lines
        xs = np.pad(xs, ((0, padding_cols), (0, 0)), mode='constant', constant_values=0)
        diffs = np.pad(diffs, ((0, padding_cols), (0, 0)), mode='constant', constant_values=0)
        
        valid_flag = np.ones(self.max_num_lines, dtype=np.int8)[:, np.newaxis]
        valid_flag[num_lines:] = 0  # from nth line to the end, set to 0

        flag_diffs = np.concatenate([valid_flag, diffs], axis=-1)
        
        xs = torch.from_numpy(xs).to(torch.float32)
        flag_diffs = torch.from_numpy(flag_diffs).to(torch.long)        

        curveset = {
            'xs': xs,
            'flag_diffs': flag_diffs, 
        }
        
        if self.sample:
            # folder_name = sample_path.split('/')[-2]
            uuid = get_filename_wo_ext(sample_path)
            curveset['uid'] = f'{uuid}'

        return curveset


class CurveDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = '',
        transform = curve_yz_scale,
        is_train: bool = True,
        replication: int = 1,
    ):
        super().__init__()
        self.transform = transform
        self.is_train = is_train
        self.replica = replication
        self.data_path = dataset_file_path
        
        self.data = self._load_data()

    def _load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        
        print(f'load {data.shape[0]} data')
        
        return data

    def __len__(self):
        if self.is_train:
            return len(self.data) * self.replica
        else:
            return len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        vertices = self.data[idx]
        
        if self.transform is not None:
            vertices = self.transform(vertices)

        vertices = torch.from_numpy(vertices).to(torch.float32)

        return vertices
    

class WireframeNormDataset(Dataset):
    def __init__(
        self, 
        dataset_path = '', 
        correct_norm_curves = False,
        shuffle_data = False,
    ):
        super().__init__()
        self.correct_norm_curves = correct_norm_curves
        self.shuffle_data = shuffle_data
        self.dataset = self._load_data(dataset_path)

    def _load_data(self, dataset_path):
        src_file_path_list_json = Path(dataset_path).joinpath('src_file_path_list.json')
        file_path_list = get_or_create_file_list_json(dataset_path, json_path=src_file_path_list_json, extension='.npz')
        
        if self.shuffle_data:
            random.shuffle(file_path_list)
        return file_path_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_path = self.dataset[idx]        
        sample = np.load(sample_path)

        uuid = get_filename_wo_ext(sample_path)
        norm_curves = sample['norm_curves']
        vertices = sample['vertices']
        adjs = sample['adjs']        

        if self.correct_norm_curves:
            norm_curves = gaussian_smooth_curve(norm_curves)

        num_curves = norm_curves.shape[0]

        norm_curves = torch.from_numpy(norm_curves).to(torch.float32)

        sample = {
            'uid': uuid, 
            'num_curves': num_curves,
            'vertices': vertices,
            'adjs': adjs,
            'norm_curves': norm_curves
        }        

        return sample