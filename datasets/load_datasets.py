from torch.utils.data import Dataset
import pandas as pd
import h5py
import numpy as np
import torch

class h5file_Dataset(Dataset):
    def __init__(self, csv_path, h5_file_path, split, val_split_ratio=0.1, seed=42):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            h5_file_path (string): Path to the single h5 file with features.
            split (string): 'train', 'val', or 'test'.
            val_split_ratio (float): The ratio of the original training set to use for validation.
            seed (int): Random seed for creating a reproducible train/val split.
        """
        self.h5_file_path = h5_file_path
        all_data = pd.read_csv(csv_path)

        # If the requested split is 'test', use the 'test' rows from the CSV
        if split == 'test':
            self.slide_data = all_data[all_data['split'] == 'test'].reset_index(drop=True)
        
        # For 'train' or 'val', we split the original 'train' data from the CSV
        else:
            # 1. Filter the CSV for all data marked as 'train'
            train_val_data = all_data[all_data['split'] == 'train'].reset_index(drop=True)
            
            # 2. Create a reproducible shuffle of the data
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(train_val_data))
            
            # 3. Define the split point (e.g., at 90% for a 9:1 split)
            split_point = int(len(shuffled_indices) * (1 - val_split_ratio))
            
            # 4. Select the correct indices based on the requested split ('train' or 'val')
            if split == 'train':
                selected_indices = shuffled_indices[:split_point]
                self.slide_data = train_val_data.iloc[selected_indices].reset_index(drop=True)
                print(f"--- Created training set with {len(self.slide_data)} samples ---")
            
            elif split == 'val':
                selected_indices = shuffled_indices[split_point:]
                self.slide_data = train_val_data.iloc[selected_indices].reset_index(drop=True)
                print(f"--- Created validation set with {len(self.slide_data)} samples ---")


    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        # Get the slide ID and label for the given index
        slide_id = self.slide_data.loc[idx, 'slide_id']
        label = self.slide_data.loc[idx, 'label']

        # Remove the '.tif' extension to match the H5 file key
        slide_id_key = slide_id.replace('.tif', '')

        # Open the H5 file and get the data for the specific slide
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            features = np.array(h5_file[slide_id_key]['feat'])
            coords = np.array(h5_file[slide_id_key]['coords'])

        return torch.from_numpy(coords), torch.from_numpy(features), label