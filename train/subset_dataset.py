import copy
import torch.utils.data as tud
import torch
import numpy as np

class SubsetStateActionDataset(tud.Dataset):
    """
    Wrap a LeRobot dataset and keep only a subset of dims in:
      - observation.state
      - action

    It also updates dataset.meta.features and dataset.meta.stats
    so that policy & preprocessor see the correct shapes.
    """

    def __init__(self, base_dataset, state_keep_names, action_keep_names):
        super().__init__()
        self.base = base_dataset

        # 1) Read original names order from meta.features
        state_feat = self.base.meta.features["observation.state"]
        action_feat = self.base.meta.features["action"]

        state_names_full = state_feat["names"]
        action_names_full = action_feat["names"]

        # 2) Compute indices based on names
        self.state_indices = [state_names_full.index(n) for n in state_keep_names]
        self.action_indices = [action_names_full.index(n) for n in action_keep_names]

        # 3) Copy meta and update features & stats
        self.meta = copy.deepcopy(self.base.meta)

        # --- Update features ---
        self.meta.features["observation.state"]["names"] = list(state_keep_names)
        self.meta.features["observation.state"]["shape"] = [len(state_keep_names)]

        self.meta.features["action"]["names"] = list(action_keep_names)
        self.meta.features["action"]["shape"] = [len(action_keep_names)]

        # -- Update stats (if exists, usually mean/std/min/max etc.) ---
        if hasattr(self.meta, "stats") and self.meta.stats is not None:
            stats = self.meta.stats

            if "observation.state" in stats:
                full_dim_state = len(state_names_full)
                for k, v in list(stats["observation.state"].items()):
                    # Convert torch / np / list to np to check the last dimension length
                    if isinstance(v, torch.Tensor):
                        arr = v.cpu().numpy()
                        is_tensor = True
                    else:
                        arr = np.asarray(v)
                        is_tensor = False

                    # Only slice when the last dimension length == full_dim_state, otherwise (e.g., count=(1,)) keep original
                    if arr.ndim > 0 and arr.shape[-1] == full_dim_state:
                        arr = arr[..., self.state_indices]
                        if is_tensor:
                            stats["observation.state"][k] = torch.from_numpy(arr).to(v.dtype)
                        else:
                            stats["observation.state"][k] = arr
                    else:
                        # For example, count=(1,), keep original
                        stats["observation.state"][k] = v

            if "action" in stats:
                full_dim_action = len(action_names_full)
                for k, v in list(stats["action"].items()):
                    if isinstance(v, torch.Tensor):
                        arr = v.cpu().numpy()
                        is_tensor = True
                    else:
                        arr = np.asarray(v)
                        is_tensor = False

                    if arr.ndim > 0 and arr.shape[-1] == full_dim_action:
                        arr = arr[..., self.action_indices]
                        if is_tensor:
                            stats["action"][k] = torch.from_numpy(arr).to(v.dtype)
                        else:
                            stats["action"][k] = arr
                    else:
                        stats["action"][k] = v

        # Print info
        print(f"SubsetStateActionDataset: original state dim {len(state_names_full)} -> {len(state_keep_names)}")
        print(f"SubsetStateActionDataset: original action dim {len(action_names_full)} -> {len(action_keep_names)}")
        print(f"meta observation.state shape: {self.meta.features['observation.state']}")
        print(f"meta action shape: {self.meta.features['action']}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        # observation.state: shape 
        if "observation.state" in sample:
            state = sample["observation.state"]
            # Support both torch.Tensor / np.ndarray, both can use ... and index list
            sample["observation.state"] = state[..., self.state_indices]

        # action: shape 
        if "action" in sample:
            action = sample["action"]
            sample["action"] = action[..., self.action_indices]

        return sample

    @property
    def num_frames(self):
        return self.base.num_frames

    @property
    def num_episodes(self):
        return self.base.num_episodes
