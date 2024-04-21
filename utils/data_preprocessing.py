import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

# Helper functions for loading and preprocessing
def load_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError("Not a valid OFF header")
        n_verts, n_faces, _ = map(int, file.readline().strip().split())
        vertices = [list(map(float, file.readline().strip().split())) for _ in range(n_verts)]
        return np.array(vertices)

def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.linalg.norm(points, axis=1))
    points /= furthest_distance
    return points

def sample_point_cloud(points, n_points=1024):
    if len(points) < n_points:
        indices = np.random.choice(len(points), n_points, replace=True)
    else:
        indices = np.random.choice(len(points), n_points, replace=False)
    return points[indices]

# Custom Dataset
class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', n_points=1024):
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.categories = [category for category in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, category))]
        self.files = []
        self.labels = []
        for label, category in enumerate(self.categories):
            new_dir = os.path.join(root_dir, category, split)
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    self.files.append(os.path.join(new_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        point_cloud = load_off(self.files[idx])
        point_cloud = normalize_point_cloud(point_cloud)
        point_cloud = sample_point_cloud(point_cloud, self.n_points)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float)
        point_cloud = point_cloud.transpose(0, 1)
        return point_cloud, self.labels[idx]

