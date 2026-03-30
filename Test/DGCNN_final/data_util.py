import os
import numpy as np
import json
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from plyfile import PlyData
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import pickle

# Function to set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_class_distribution(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    class_counts = Counter(all_labels)
    return class_counts

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    N = point.shape[0]
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def load_ply_file(file_path):
    plydata = PlyData.read(file_path)
    pc = np.vstack([plydata['vertex'][axis] for axis in ['x', 'y', 'z']]).T
    return pc

def augment_data(point_set, rotate=True, add_noise=True):
    if rotate:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        point_set[:, :3] = point_set[:, :3].dot(rotation_matrix)

    if add_noise:
        noise = np.random.normal(0, 0.02, point_set[:, :3].shape)
        point_set[:, :3] += noise

    return point_set

def augment_point_cloud(point_set, rotate=True, add_noise=False):
    if rotate:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        point_set[:, :3] = point_set[:, :3].dot(rotation_matrix)

    if add_noise:
        noise = np.random.normal(0, 0.02, point_set[:, :3].shape)
        point_set[:, :3] += noise

    return point_set

def is_preprocessed(processed_root, split):
    split_root = os.path.join(processed_root, split)
    return os.path.exists(split_root) and len(os.listdir(split_root)) > 0

class CustomDataLoader(Dataset):
    def __init__(self, root, split='train', npoints=1024, use_normals=False, uniform=False, normalize=True, rotate=True, add_noise=False, class_names=None):
        """
        Custom DataLoader for preprocessed point cloud data.

        Args:
            root (str): Root directory containing subfolders 'train', 'val', 'test'.
            split (str): Dataset split to load ('train', 'val', or 'test').
            npoints (int): Number of points to sample from each point cloud.
            use_normals (bool): Whether to include normals in the point cloud.
            uniform (bool): Whether to use farthest point sampling.
            normalize (bool): Whether to normalize the point cloud.
            rotate (bool): Whether to apply random rotations.
            add_noise (bool): Whether to add random noise.
            class_names (dict): Dictionary mapping class names to integer labels.
        """
        self.root = root
        self.split = split
        self.npoints = npoints
        self.use_normals = use_normals
        self.uniform = uniform
        self.normalize = normalize
        self.rotate = rotate
        self.add_noise = add_noise

        # Validate class names
        if class_names is not None:
            self.classes = class_names
        else:
            raise ValueError("Class names must be provided. Example: {'Discoide': 0, 'Levallois': 1, 'Laminaire': 2}")

        # Validate split directory
        split_root = os.path.join(root, split)
        if not os.path.exists(split_root):
            raise ValueError(f"Split directory '{split_root}' does not exist.")

        print(f"Contents of split directory ({split_root}): {os.listdir(split_root)}")
        print(f"Initialized classes: {self.classes}")

        # Load datapath
        self.datapath = []
        for file in os.listdir(split_root):
            if file.endswith('_points.npy') and not file.startswith('.'):
                flake_id = file.replace('_points.npy', '')
                points_path = os.path.join(split_root, f"{flake_id}_points.npy")
                label_path = os.path.join(split_root, f"{flake_id}_label.npy")
                if os.path.exists(points_path) and os.path.exists(label_path):
                    self.datapath.append((points_path, label_path))
                else:
                    raise ValueError(f"Missing corresponding label file for {points_path}")

        print(f"The size of {split} data is {len(self.datapath)}")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        points_path, label_path = self.datapath[index]

        # Load point cloud and label
        point_set = np.load(points_path)
        label = np.load(label_path)

        # Sample points if necessary
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[:self.npoints, :]

        # Normalize and augment data
        if self.normalize:
            point_set[:, :3] = pc_normalize(point_set[:, :3])
        point_set = augment_data(point_set, rotate=self.rotate, add_noise=self.add_noise)

        # Exclude normals if not required
        if not self.use_normals:
            point_set = point_set[:, :3]

        return point_set, label[0]


class KFoldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        processed_root,
        class_names,
        label_dict,
        npoints=1024,
        track_flake_id=True,
        process_already=True,
        normalize=True,
        train_ratio=0.9,
        val_ratio=0.0,
        test_ratio=0.1,
        split_seed=42,
        rotate=False,
        add_noise=False
    ):
        self.root = root
        self.processed_root = processed_root
        self.class_names = class_names
        self.label_dict = label_dict
        self.npoints = npoints
        self.track_flake_id = track_flake_id
        self.process_already = process_already
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.rotate = rotate
        self.add_noise = add_noise

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        if not os.path.exists(self.processed_root):
            os.makedirs(self.processed_root)

        if not self.process_already:
            self._process_ply_files_for_kfold()

        self.data_list = []
        self._load_kfold_data()

    def _process_ply_files_for_kfold(self):
        if self.class_names is None or self.label_dict is None:
            raise ValueError("class_names and label_dict must be provided when process_already=False")

        class_files = {name: [] for name in self.class_names.keys()}
        skipped = []

        ply_files = [f for f in os.listdir(self.root) if f.endswith('.ply')]
        for file_name in tqdm(ply_files, desc="Categorizing files"):
            file_path = os.path.join(self.root, file_name)
            flake_id = os.path.splitext(file_name)[0]

            if flake_id not in self.label_dict:
                skipped.append((flake_id, 'not in label_dict'))
                continue

            label_name = self.label_dict[flake_id]
            if label_name not in self.class_names:
                skipped.append((flake_id, f"class '{label_name}' not in class_names"))
                continue

            class_files[label_name].append((flake_id, file_path))

        splits = {'train': [], 'test': []}
        for class_name, files in class_files.items():
            if not files:
                continue

            if self.test_ratio > 0:
                kfold_files, test_files = train_test_split(
                    files,
                    test_size=self.test_ratio,
                    random_state=self.split_seed
                )
                splits['train'].extend(kfold_files)
                splits['test'].extend(test_files)
            else:
                splits['train'].extend(files)

        for split_name, file_list in splits.items():
            if not file_list:
                continue

            split_root = os.path.join(self.processed_root, split_name)
            os.makedirs(split_root, exist_ok=True)

            for flake_id, file_path in tqdm(file_list, desc=f"Processing {split_name}"):
                try:
                    label_name = self.label_dict[flake_id]
                    label_num = self.class_names[label_name]

                    point_set = load_ply_file(file_path)
                    if self.normalize:
                        point_set[:, :3] = pc_normalize(point_set[:, :3])

                    points_path = os.path.join(split_root, f"{flake_id}_points.npy")
                    label_path = os.path.join(split_root, f"{flake_id}_label.npy")

                    np.save(points_path, point_set.astype(np.float32))
                    np.save(label_path, np.array([label_num], dtype=np.int64))
                except Exception as exc:
                    print(f"Error processing {flake_id}: {exc}")

    def _load_kfold_data(self):
        self.data_list.clear()
        train_dir = os.path.join(self.processed_root, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Processed train directory missing: {train_dir}")

        point_files = [f for f in os.listdir(train_dir) if f.endswith('_points.npy')]
        for point_file in point_files:
            flake_id = point_file.replace('_points.npy', '')
            point_path = os.path.join(train_dir, point_file)
            label_path = os.path.join(train_dir, f"{flake_id}_label.npy")

            if not os.path.exists(label_path):
                print(f"Warning: missing label for {flake_id}")
                continue

            try:
                label = np.load(label_path).item()
            except Exception as exc:
                print(f"Failed to load label for {flake_id}: {exc}")
                continue

            self.data_list.append({
                'point_path': point_path,
                'label_path': label_path,
                'flake_id': flake_id,
                'label': label
            })

    def _print_class_distribution(self):
        if not self.data_list:
            print("No samples loaded for k-fold dataset")
            return

        counts = Counter(item['label'] for item in self.data_list)
        reverse_map = {v: k for k, v in self.class_names.items()}
        total = len(self.data_list)
        for label_idx, count in sorted(counts.items()):
            class_name = reverse_map.get(label_idx, f"class_{label_idx}")
            pct = (count / total) * 100
            print(f"  - {class_name}: {count} samples ({pct:.1f}%)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        try:
            points = np.load(item['point_path']).astype(np.float32)
            label = np.load(item['label_path']).item()

            if len(points) > self.npoints:
                points = farthest_point_sample(points, self.npoints)
            elif len(points) < self.npoints:
                idx = np.random.choice(len(points), self.npoints, replace=True)
                points = points[idx]

            if points.shape[1] > 3:
                points = points[:, :3]

            if self.rotate or self.add_noise:
                points = augment_point_cloud(points.copy(), rotate=self.rotate, add_noise=self.add_noise)

            tensor_points = torch.from_numpy(points.astype(np.float32))
            tensor_label = torch.tensor(int(label), dtype=torch.long)

            if self.track_flake_id:
                return tensor_points, tensor_label, item['flake_id']
            return tensor_points, tensor_label
        except Exception as exc:
            print(f"Error loading sample {index} (flake {item.get('flake_id', 'unknown')}): {exc}")
            dummy_points = torch.zeros((self.npoints, 3), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            if self.track_flake_id:
                return dummy_points, dummy_label, f"error_{index}"
            return dummy_points, dummy_label

    def get_test_dataset(self):
        """Return a Dataset for the held-out test split if it exists."""
        test_dir = os.path.join(self.processed_root, 'test')
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return None

        point_files = [f for f in os.listdir(test_dir) if f.endswith('_points.npy')]
        if not point_files:
            print(f"No held-out test samples located under {test_dir}")
            return None

        class _HeldOutTestDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, root_dir, files):
                self.base = base_dataset
                self.root_dir = root_dir
                self.entries = []
                for point_file in sorted(files):
                    flake_id = point_file.replace('_points.npy', '')
                    point_path = os.path.join(root_dir, point_file)
                    label_path = os.path.join(root_dir, f"{flake_id}_label.npy")
                    if os.path.exists(label_path):
                        self.entries.append((flake_id, point_path, label_path))
                    else:
                        print(f"Warning: missing label file for test sample {flake_id}")

            def __len__(self):
                return len(self.entries)

            def __getitem__(self, idx):
                flake_id, point_path, label_path = self.entries[idx]
                points = np.load(point_path).astype(np.float32)
                if points.shape[0] > self.base.npoints:
                    points = farthest_point_sample(points, self.base.npoints)
                elif points.shape[0] < self.base.npoints:
                    indices = np.random.choice(points.shape[0], self.base.npoints, replace=True)
                    points = points[indices]

                if points.shape[1] > 3:
                    points = points[:, :3]

                if self.base.normalize:
                    points = pc_normalize(points)

                tensor_points = torch.from_numpy(points.astype(np.float32))

                raw_label = np.load(label_path)
                if isinstance(raw_label, np.ndarray):
                    label_value = int(raw_label.flatten()[0])
                else:
                    label_value = int(raw_label)
                tensor_label = torch.tensor(label_value, dtype=torch.long)

                if self.base.track_flake_id:
                    return tensor_points, tensor_label, flake_id
                return tensor_points, tensor_label

        return _HeldOutTestDataset(self, test_dir, point_files)
