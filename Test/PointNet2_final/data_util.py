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


def analyze_skipped_files(root, label_dict, class_names, csv_path=None):
    """
    Analyze which .ply files would be skipped during processing and why.
    
    Args:
        root (str): Path to .ply files
        label_dict (dict): Flake ID to class name mapping
        class_names (dict): Class name to integer mapping
        csv_path (str, optional): Path to CSV for additional checks
    
    Returns:
        dict: Analysis of skipped files
    """
    # Get all .ply files
    ply_files = []
    for file in os.listdir(root):
        if file.endswith('.ply'):
            flake_id = os.path.splitext(file)[0]
            ply_files.append(flake_id)
    
    print(f"Found {len(ply_files)} .ply files in {root}")
    
    # Analyze each file
    processed = []
    skipped = []
    
    for flake_id in ply_files:
        if flake_id in label_dict:
            label_name = label_dict[flake_id]
            if label_name in class_names:
                processed.append({
                    'flake_id': flake_id,
                    'class': label_name,
                    'reason': 'processed'
                })
            else:
                skipped.append({
                    'flake_id': flake_id,
                    'reason': f'class "{label_name}" not in class_names',
                    'class': label_name
                })
        else:
            skipped.append({
                'flake_id': flake_id,
                'reason': 'not in label_dict',
                'class': None
            })
    
    # Summary
    print(f"Would process: {len(processed)} files")
    print(f"Would skip: {len(skipped)} files")
    
    if skipped:
        print("\nReasons for skipping:")
        reasons = {}
        for item in skipped:
            reason = item['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        
        for reason, count in reasons.items():
            print(f"- {reason}: {count}")
        
        print("\nFirst 10 skipped files:")
        for item in skipped[:10]:
            print(f"- {item['flake_id']}: {item['reason']}")
    
    # Check CSV if provided
    csv_analysis = None
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            csv_flake_ids = set(df['Artefact_ID'].astype(str))
            in_ply_not_csv = set(ply_files) - csv_flake_ids
            in_csv_not_ply = csv_flake_ids - set(ply_files)
            
            csv_analysis = {
                'csv_total': len(csv_flake_ids),
                'in_ply_not_csv': list(in_ply_not_csv),
                'in_csv_not_ply': list(in_csv_not_ply)
            }
            
            print(f"\nCSV Analysis:")
            print(f"CSV entries: {len(csv_flake_ids)}")
            print(f"In .ply folder but not CSV: {len(in_ply_not_csv)}")
            print(f"In CSV but not .ply folder: {len(in_csv_not_ply)}")
            
            if in_ply_not_csv:
                print("Files in .ply folder but not in CSV:")
                for fid in list(in_ply_not_csv)[:5]:
                    print(f"- {fid}")
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    return {
        'total_ply': len(ply_files),
        'processed': processed,
        'skipped': skipped,
        'csv_analysis': csv_analysis
    }

def verify_label_dict_with_csv(csv_path, label_dict):
    """
    Verify that the label_dict matches the original CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        label_dict (dict): Dictionary to verify.
    
    Returns:
        dict: Verification results.
    """
    df = pd.read_csv(csv_path)
    csv_dict = dict(zip(df['Artefact_ID'], df['Broad_Strategy']))
    
    mismatches = []
    for flake_id, label in label_dict.items():
        if flake_id not in csv_dict:
            mismatches.append({
                'flake_id': flake_id,
                'issue': 'Flake ID in label_dict but not in CSV'
            })
        elif csv_dict[flake_id] != label:
            mismatches.append({
                'flake_id': flake_id,
                'label_dict_value': label,
                'csv_value': csv_dict[flake_id]
            })
    
    # Check for missing in label_dict
    missing_in_dict = [fid for fid in csv_dict if fid not in label_dict]
    
    result = {
        'csv_entries': len(csv_dict),
        'dict_entries': len(label_dict),
        'mismatches': mismatches,
        'missing_in_dict': missing_in_dict,
        'status': 'Matches CSV!' if not mismatches and not missing_in_dict else f'Issues found: {len(mismatches)} mismatches, {len(missing_in_dict)} missing'
    }
    
    print(f"CSV verification: {len(csv_dict)} entries in CSV, {len(label_dict)} in dict.")
    if mismatches or missing_in_dict:
        print(f"Issues: {len(mismatches)} mismatches, {len(missing_in_dict)} missing in dict.")
    else:
        print("Label dict matches CSV perfectly!")
    
    return result

def count_samples_by_class(train_dataset_folder):
    file_names = os.listdir(train_dataset_folder)
    
    # Filter files ending with '_points.npy'
    points_files = [file for file in file_names if file.endswith('_points.npy')]
    
    # Extract class labels from file names
    class_labels = []
    for file in points_files:
        parts = file.split('_')
        class_label = parts[3]  # Extract the class label between the third and fourth underscore
        class_labels.append(class_label)
    
    # Count samples by class
    class_counts = Counter(class_labels)
    
    return class_counts

# Load class names from metadata file
def load_class_names(meta_file):
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    return metadata['class_names']

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

class CustomDataLoader_with_labels(Dataset):
    def __init__(self, root, processed_root=None, split='train', npoints=1024, use_normals=False, normalize=True, rotate=True, add_noise=False, class_names=None, label_dict=None, process_already=True, track_flake_id=False, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, split_seed=42):
        """
        Custom DataLoader for PointNet classification with label mapping and optional processing.

        Args:
            root (str): Path to the root directory. If process_already=False, this contains .ply files. If process_already=True, this is ignored and processed_root is used.
            processed_root (str): Path to save/load preprocessed .npy files. If None, defaults to root.
            split (str): Dataset split ('train', 'val', or 'test').
            npoints (int): Number of points to sample from each point cloud.
            use_normals (bool): Whether to use normals (if available).
            normalize (bool): Whether to normalize the point cloud.
            rotate (bool): Whether to apply random rotation augmentation.
            add_noise (bool): Whether to add noise to the point cloud.
            class_names (dict): Mapping of class names to integers, e.g., {'Discoide': 0, 'Levallois': 1, 'Laminaire': 2}.
            label_dict (dict): Mapping of flake IDs to class names, loaded from CSV.
            process_already (bool): If False, process and split .ply files to .npy. If True, load from existing .npy.
            track_flake_id (bool): If True, return flake_id along with point_set and label.
            train_ratio (float): Proportion of data for training (default 0.7).
            val_ratio (float): Proportion of data for validation (default 0.2).
            test_ratio (float): Proportion of data for testing (default 0.1). Must sum to 1.0.
            split_seed (int): Random seed for reproducible train/val/test splitting (default 42).
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

        self.root = root
        self.processed_root = processed_root if processed_root is not None else root
        self.split = split
        self.npoints = npoints
        self.use_normals = use_normals
        self.normalize = normalize
        self.rotate = rotate
        self.add_noise = add_noise
        self.class_names = class_names
        self.label_dict = label_dict
        self.process_already = process_already
        self.track_flake_id = track_flake_id
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        split_root = os.path.join(self.processed_root, split)
        if not os.path.exists(split_root):
            os.makedirs(split_root)

        if not self.process_already:
            if self.class_names is None or self.label_dict is None:
                raise ValueError("class_names and label_dict must be provided when process_already=False")
            self._process_and_split_ply_files()

        # Load from .npy files
        self.datapath = []
        self.flake_ids = []
        for file in sorted(os.listdir(split_root)):
            if file.endswith('_points.npy'):
                base_name = file.replace('_points.npy', '')
                points_path = os.path.join(split_root, f"{base_name}_points.npy")
                label_path = os.path.join(split_root, f"{base_name}_label.npy")
                if os.path.exists(points_path) and os.path.exists(label_path):
                    self.datapath.append((points_path, label_path))
                    if self.track_flake_id:
                        self.flake_ids.append(base_name)
                else:
                    raise ValueError(f"Missing corresponding label file for {points_path}")

        print(f"Loaded {len(self.datapath)} samples for split '{split}'.")

        self._check_npoints()

    def _process_and_split_ply_files(self):
        """
        Process .ply files: group by class, split into train/val/test, and save as .npy in respective folders.
        """
        # Collect files per class
        class_files = {class_name: [] for class_name in self.class_names.keys()}
        for file in os.listdir(self.root):
            if file.endswith('.ply'):
                file_path = os.path.join(self.root, file)
                flake_id = os.path.splitext(file)[0]
                label_name = self.label_dict.get(flake_id)
                if label_name and label_name in self.class_names:
                    class_files[label_name].append((flake_id, file_path))

        # Split each class
        splits = {'train': [], 'val': [], 'test': []}
        for class_name, files in class_files.items():
            if not files:
                continue
            # Split: train vs temp (val+test)
            train_files, temp_files = train_test_split(files, test_size=self.val_ratio + self.test_ratio, random_state=self.split_seed)
            # Split temp into val and test
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_files, test_files = train_test_split(temp_files, test_size=1 - val_ratio_adjusted, random_state=self.split_seed)
            splits['train'].extend(train_files)
            splits['val'].extend(val_files)
            splits['test'].extend(test_files)

        # Process and save for each split
        for split_name, file_list in splits.items():
            split_root = os.path.join(self.processed_root, split_name)
            if not os.path.exists(split_root):
                os.makedirs(split_root)
            for flake_id, file_path in file_list:
                label_name = self.label_dict[flake_id]
                label_num = self.class_names[label_name]
                point_set = load_ply_file(file_path)
                if self.normalize:
                    point_set[:, :3] = pc_normalize(point_set[:, :3])
                points_path = os.path.join(split_root, f"{flake_id}_points.npy")
                label_path = os.path.join(split_root, f"{flake_id}_label.npy")
                np.save(points_path, point_set)
                np.save(label_path, np.array([label_num]))

    def _check_npoints(self):
        """
        Check if the specified npoints is valid for the dataset.
        """
        for points_path, _ in self.datapath:
            points = np.load(points_path)
            if self.npoints > points.shape[0]:
                raise ValueError(f"Specified npoints ({self.npoints}) is greater than the number of points ({points.shape[0]}) in file {points_path}.")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        points_path, label_path = self.datapath[index]

        point_set = np.load(points_path)
        label = np.load(label_path).item()

        # Sample npoints
        if self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        # Normalize and augment
        if self.normalize:
            point_set[:, :3] = pc_normalize(point_set[:, :3])
        if self.rotate or self.add_noise:
            point_set = augment_point_cloud(point_set, rotate=self.rotate, add_noise=self.add_noise)

        # Use only XYZ if normals are not used
        if not self.use_normals:
            point_set = point_set[:, :3]

        if self.track_flake_id:
            flake_id = self.flake_ids[index]
            return point_set, label, flake_id
        else:
            return point_set, label

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class KFoldDataset(torch.utils.data.Dataset):
    def __init__(self, root, processed_root, class_names, label_dict,
                 npoints=1024, track_flake_id=True, process_already=True,
                 normalize=True, train_ratio=0.9, val_ratio=0.0, test_ratio=0.1,
                 split_seed=42, rotate=False, add_noise=False):
        """
        Initialize KFoldDataset for cross-validation.
        
        Args:
            train_ratio (float): Ratio of total data to use for k-fold CV
            val_ratio (float): Should be 0.0 for k-fold (handled internally by StratifiedKFold)
            test_ratio (float): Ratio of total data to hold out for final testing
            rotate (bool): Whether to apply random rotation augmentation per sample (default False)
            add_noise (bool): Whether to add Gaussian noise during augmentation (default False)
            
        Note: train_ratio + test_ratio should sum to 1.0
              val_ratio should be 0.0 (k-fold handles validation internally)
        """
        self.root = root
        self.processed_root = processed_root
        self.class_names = class_names
        self.label_dict = label_dict
        self.npoints = npoints
        self.track_flake_id = track_flake_id
        self.process_already = process_already
        self.normalize = normalize
        self.train_ratio = train_ratio  # For k-fold CV portion
        self.val_ratio = val_ratio      # Should be 0.0 
        self.test_ratio = test_ratio    # For held-out test set
        self.split_seed = split_seed
        self.rotate = rotate
        self.add_noise = add_noise
        
        # ✅ CORRECTED: Validate ratios for k-fold with held-out test set
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must sum to 1.0")
        
        if val_ratio != 0.0:
            print(f"⚠️  Warning: val_ratio should be 0.0 for k-fold CV (k-fold handles validation). Got {val_ratio}")
        
        # Ensure processed_root exists
        if not os.path.exists(self.processed_root):
            os.makedirs(self.processed_root)
        
        # Process data if needed
        if not self.process_already:
            print("Processing .ply files for k-fold cross-validation...")
            self._process_ply_files_for_kfold()
        
        # Load all available data for k-fold CV
        self.data_list = []
        self._load_kfold_data()
        
        print(f"KFoldDataset initialized with {len(self.data_list)} samples for cross-validation")
        print(f"Test set ratio: {test_ratio:.1%} (excluded from k-fold)")
        self._print_class_distribution()
    
    def _process_ply_files_for_kfold(self):
        """
        Process .ply files and create proper train/test splits for k-fold CV.
        """
        if self.class_names is None or self.label_dict is None:
            raise ValueError("class_names and label_dict must be provided when process_already=False")
        
        # Collect files per class
        class_files = {class_name: [] for class_name in self.class_names.keys()}
        skipped_files = []
        
        print("Scanning .ply files...")
        ply_files = [f for f in os.listdir(self.root) if f.endswith('.ply')]
        
        for file in tqdm(ply_files, desc="Categorizing files"):
            file_path = os.path.join(self.root, file)
            flake_id = os.path.splitext(file)[0]
            
            if flake_id not in self.label_dict:
                skipped_files.append((flake_id, "not in label_dict"))
                continue
            
            label_name = self.label_dict[flake_id]
            if label_name not in self.class_names:
                skipped_files.append((flake_id, f"class '{label_name}' not in class_names"))
                continue
            
            class_files[label_name].append((flake_id, file_path))
        
        # Print statistics
        total_valid = sum(len(files) for files in class_files.values())
        print(f"\nFile processing summary:")
        print(f"Total .ply files: {len(ply_files)}")
        print(f"Valid files: {total_valid}")
        print(f"Skipped files: {len(skipped_files)}")
        
        # ✅ CORRECTED: Proper splitting for k-fold with held-out test set
        splits = {'train': [], 'test': []}  # Only train (for k-fold) and test (held-out)
        
        for class_name, files in class_files.items():
            if not files:
                print(f"Warning: No files found for class '{class_name}'")
                continue
            
            if self.test_ratio > 0:
                # Split: k-fold portion vs held-out test set
                kfold_files, test_files = train_test_split(
                    files, 
                    test_size=self.test_ratio, 
                    random_state=self.split_seed, 
                    stratify=None
                )
                splits['test'].extend(test_files)
                splits['train'].extend(kfold_files)  # All k-fold data goes to 'train'
                
                print(f"Class {class_name}: {len(kfold_files)} for k-fold, {len(test_files)} for test")
            else:
                # No test set, all data for k-fold
                splits['train'].extend(files)
                print(f"Class {class_name}: {len(files)} for k-fold (no test set)")
        
        # Process and save files for each split
        for split_name, file_list in splits.items():
            if not file_list:
                continue
                
            split_root = os.path.join(self.processed_root, split_name)
            if not os.path.exists(split_root):
                os.makedirs(split_root)
            
            print(f"\nProcessing {len(file_list)} files for {split_name} split...")
            
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
                    
                except Exception as e:
                    print(f"Error processing {flake_id}: {e}")
                    continue
        
        print(f"\n✅ Preprocessing completed!")
        print(f"K-fold data: {len(splits.get('train', []))} samples")
        print(f"Held-out test: {len(splits.get('test', []))} samples")
        print(f"Files saved to: {self.processed_root}")
    
    def _load_kfold_data(self):
        """
        Load ONLY the train data for k-fold CV.
        Test data is completely excluded from k-fold cross-validation.
        """
        # ✅ CORRECTED: Only load 'train' split for k-fold
        splits_to_load = ['train']  # Exclude 'test' from k-fold
        
        for split_name in splits_to_load:
            split_dir = os.path.join(self.processed_root, split_name)
            if not os.path.exists(split_dir):
                print(f"Warning: {split_name} directory not found: {split_dir}")
                continue
            
            point_files = [f for f in os.listdir(split_dir) if f.endswith('_points.npy')]
            
            for point_file in point_files:
                flake_id = point_file.replace('_points.npy', '')
                point_path = os.path.join(split_dir, point_file)
                label_path = os.path.join(split_dir, f"{flake_id}_label.npy")
                
                if os.path.exists(point_path) and os.path.exists(label_path):
                    try:
                        label = np.load(label_path).item()
                        if self.label_dict and flake_id in self.label_dict:
                            expected_label = self.class_names[self.label_dict[flake_id]]
                            if label != expected_label:
                                print(f"Warning: Label mismatch for {flake_id}: "
                                     f"file={label}, expected={expected_label}")
                        
                        self.data_list.append({
                            'point_path': point_path,
                            'label_path': label_path,
                            'flake_id': flake_id,
                            'label': label,
                            'split_origin': split_name
                        })
                    except Exception as e:
                        print(f"Error loading label for {flake_id}: {e}")
                else:
                    print(f"Warning: Missing files for {flake_id}")
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the k-fold dataset."""
        if not self.data_list:
            print("No data loaded!")
            return
        
        # Count labels
        label_counts = {}
        for item in self.data_list:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Convert to class names for display
        print(f"\nClass distribution in k-fold dataset:")
        reverse_class_names = {v: k for k, v in self.class_names.items()}
        total_samples = len(self.data_list)
        
        for label, count in sorted(label_counts.items()):
            class_name = reverse_class_names.get(label, f"Unknown({label})")
            percentage = (count / total_samples) * 100
            print(f"  - {class_name}: {count} samples ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        Get a single sample for k-fold cross-validation.
        """
        item = self.data_list[index]
        
        try:
            # Load point cloud
            points = np.load(item['point_path']).astype(np.float32)
            
            # Load label
            label = np.load(item['label_path']).item()
            
            # ✅ IMPROVED: Use FPS instead of random sampling
            if len(points) > self.npoints:
                # Use Farthest Point Sampling for better coverage
                points = farthest_point_sample(points, self.npoints)
            elif len(points) < self.npoints:
                # Upsample with replacement (same as before)
                indices = np.random.choice(len(points), self.npoints, replace=True)
                points = points[indices]
            
            # Ensure we only use XYZ coordinates (no normals for now)
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # Optional augmentation (rotation without noise by default)
            if self.rotate or self.add_noise:
                points = augment_point_cloud(points.copy(), rotate=self.rotate, add_noise=self.add_noise)

            points = points.astype(np.float32, copy=False)

            # Convert to tensors
            points = torch.from_numpy(points)
            label = torch.tensor(label, dtype=torch.long)
            
            if self.track_flake_id:
                return points, label, item['flake_id']
            else:
                return points, label
                
        except Exception as e:
            print(f"Error loading sample {index} (flake_id: {item.get('flake_id', 'unknown')}): {e}")
            # Return dummy data to prevent crashes
            dummy_points = torch.zeros((self.npoints, 3), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            if self.track_flake_id:
                return dummy_points, dummy_label, f"error_{index}"
            else:
                return dummy_points, dummy_label
    
    def get_test_dataset(self):
        """
        Create a separate dataset instance for the test set (excluded from k-fold).
        Returns None if no test set was created.
        """
        test_dir = os.path.join(self.processed_root, 'test')
        if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
            print("No test set available")
            return None
        
        # Create a CustomDataLoader_with_labels instance for test data
        try:
            test_dataset = CustomDataLoader_with_labels(
                root=self.root,
                processed_root=self.processed_root,
                split='test',
                npoints=self.npoints,
                class_names=self.class_names,
                label_dict=self.label_dict,
                process_already=True,
                track_flake_id=self.track_flake_id,
                normalize=self.normalize,
                rotate=False,  # No augmentation for test
                add_noise=False
            )
            return test_dataset
        except Exception as e:
            print(f"Error creating test dataset: {e}")
            return None


