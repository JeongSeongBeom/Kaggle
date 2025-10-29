import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, df, cat_features, num_features, target):
        self.cats = torch.tensor(df[cat_features].values, dtype=torch.long)
        self.nums = torch.tensor(df[num_features].values, dtype=torch.float32)
        self.y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cats[idx], self.nums[idx], self.y[idx]

