import os
import json
import numpy as np
from PIL import Image

class EuroSATDataLoader:
    def __init__(self, root_dir, split_config="dataset_split.json", split_ratio=(0.7, 0.15, 0.15), random_seed=42):
        self.root_dir = root_dir
        self.split_config = split_config
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 核心数据存储
        self.data = {'train': {'X': [], 'y': []}, 
                     'val': {'X': [], 'y': []}, 
                     'test': {'X': [], 'y': []}}
        
        self._prepare_data()

    def _prepare_data(self):
        if os.path.exists(self.split_config):
            print(f"加载划分配置文件 {self.split_config}")
            with open(self.split_config, 'r') as f:
                split_dict = json.load(f)
        else:
            print("无划分配置文件, 尝试生成并保存")
            split_dict = self._generate_split()
            with open(self.split_config, 'w') as f:
                json.dump(split_dict, f, indent=4)
        
        for split_name in ['train', 'val', 'test']:
            print(f"正在加载 {split_name} 集数据")
            X_list, y_list = [], []
            for file_path in split_dict[split_name]:
                cls_name = file_path.split(os.sep)[-2]
                label = self.class_to_idx[cls_name]
                
                full_path = os.path.join(self.root_dir, file_path)
                img = Image.open(full_path).convert('RGB')
                
                img_array = np.array(img, dtype=np.float32).flatten()
                
                X_list.append(img_array)
                y_list.append(label)
            
            self.data[split_name]['X'] = np.vstack(X_list)
            self.data[split_name]['y'] = np.array(y_list, dtype=np.int32)
        
        self._normalize()

    def _generate_split(self):
        np.random.seed(self.random_seed)
        split_dict = {'train': [], 'val': [], 'test': []}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            file_paths = [os.path.join(cls_name, f) for f in files]
            
            np.random.shuffle(file_paths)
            
            n_total = len(file_paths)
            n_train = int(n_total * self.split_ratio[0])
            n_val = int(n_total * self.split_ratio[1])
            
            split_dict['train'].extend(file_paths[:n_train])
            split_dict['val'].extend(file_paths[n_train:n_train+n_val])
            split_dict['test'].extend(file_paths[n_train+n_val:])
            
        return split_dict

    def _normalize(self):
        print("按测试集进行数据归一化")
        # X_train shape: (N_train, 12288)
        self.mean = np.mean(self.data['train']['X'], axis=0, keepdims=True)
        self.std = np.std(self.data['train']['X'], axis=0, keepdims=True)
        
        self.std = np.clip(self.std, 1e-8, None)
        
        for split_name in ['train', 'val', 'test']:
            self.data[split_name]['X'] = (self.data[split_name]['X'] - self.mean) / self.std

    def get_batch(self, split_name, batch_size, shuffle=True):
        X = self.data[split_name]['X']
        y = self.data[split_name]['y']
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]

    def get_full_data(self, split_name):
        return self.data[split_name]['X'], self.data[split_name]['y']