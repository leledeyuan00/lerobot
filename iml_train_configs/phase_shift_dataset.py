import torch
from torch.utils.data import Dataset, ConcatDataset
import copy

class PhaseShiftedDataset(Dataset):
    def __init__(self, dataset, phase_offset):
        self.dataset = dataset
        self.phase_offset = phase_offset
        # Inherit meta from the original dataset
        # Inherit stats if available
        self.meta = copy.deepcopy(self.dataset.meta)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Fetch the original item
        item = self.dataset[idx]

        # 2. Dynamically modify the phase in the observation.state
        if 'observation.state' in item:
            # Assuming 'phase' is stored in a specific index of the state vector
            state = item['observation.state']
            # Here we assume the last element of the state vector is the phase
            state[-1] = state[-1] + self.phase_offset
            item['observation.state'] = state
            
        return item
    def __getattr__(self, name):
        # 关键：如果在当前类找不到属性（比如 num_frames, stats），
        # 就去原始 dataset 里找
        return getattr(self.dataset, name)


class MultiTaskDataset(Dataset):
    """
    用于替代 ConcatDataset 的增强版 Wrapper。
    1. 聚合多个数据集的数据。
    2. 正确计算 num_frames, num_episodes 等聚合属性。
    3. 借用主数据集的 stats, fps 等静态属性。
    """
    def __init__(self, datasets, main_stats_dataset_idx=0):
        self.datasets = datasets
        # 使用 PyTorch 原生 ConcatDataset 处理索引映射，不用自己写
        self.concat_ds = ConcatDataset(datasets)
        # 指定使用哪个数据集的 stats (通常是第0个，即挂衣服任务)
        self.main_ds = datasets[main_stats_dataset_idx]

    def __len__(self):
        return len(self.concat_ds)

    def __getitem__(self, idx):
        return self.concat_ds[idx]

    @property
    def num_frames(self):
        # 聚合属性：总帧数是所有子数据集帧数之和
        return sum(d.num_frames for d in self.datasets)

    @property
    def num_episodes(self):
        # 聚合属性：总 episode 数
        return sum(d.num_episodes for d in self.datasets)

    @property
    def meta(self):
        # 静态属性：强制使用主数据集的 stats，保证归一化标准统一
        return self.main_ds.meta


    def __getattr__(self, name):
        # 兜底：其他未定义的属性，默认从主数据集取
        return getattr(self.main_ds, name)