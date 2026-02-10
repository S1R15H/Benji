
import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import sys

# Add src to path to import preprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from env.preprocessing import BenjiPreprocessor

class BenjiBCDataset(Dataset):
    def __init__(self, data_dir="data/raw", stack_size=4):
        self.data_dir = data_dir
        self.stack_size = stack_size
        self.preprocessor = BenjiPreprocessor()
        
        self.samples = []
        self.image_cache = {} # RAM Cache
        
        # 1. Discover all sessions
        if not os.path.exists(data_dir):
            print(f"Dataset Warning: {data_dir} does not exist.")
            return

        session_dirs = sorted(glob.glob(os.path.join(data_dir, "session_*")))
        print(f"Found {len(session_dirs)} sessions.")
        
        for session_path in session_dirs:
            self._load_session(session_path)
            
        print(f"Total Samples Loaded: {len(self.samples)}")
        print("Pre-loading images into RAM...")
        self._preload_images()
        print(f"Cached {len(self.image_cache)} images.")

    def _preload_images(self):
        """Loads all unique images referenced in samples into RAM."""
        unique_paths = set()
        for s in self.samples:
            for p in s['paths']:
                if p is not None:
                    unique_paths.add(p)
        
        for p in unique_paths:
            if p not in self.image_cache:
                raw_bgr = cv2.imread(p)
                if raw_bgr is not None:
                    frame = self.preprocessor.process_frame(raw_bgr)
                    # Frame is (1, 128, 128) - CHW
                    # Squeeze channel dim: (1, 128, 128) -> (128, 128)
                    if frame.ndim == 3 and frame.shape[0] == 1:
                        frame = frame[0, :, :]
                    self.image_cache[p] = frame
                else:
                    # corrupted or missing
                    self.image_cache[p] = np.zeros((128, 128), dtype=np.uint8)

    def _load_session(self, session_path):
        """Parses actions.csv and verifies frame existence."""
        csv_path = os.path.join(session_path, "actions.csv")
        frames_dir = os.path.join(session_path, "frames")
        
        if not os.path.exists(csv_path):
            return
            
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            # We need to store everything in memory or just indices?
            # Storing indices is memory efficient.
            # But we need random access for stacking.
            
            # Read all rows first strictly sorted by frame_id
            rows = list(reader)
            rows.sort(key=lambda x: int(x['frame_id']))
            
            # Convert to list of dicts for faster access
            session_data = []
            for row in rows:
                frame_id = int(row['frame_id'])
                action = int(row['action'])
                img_path = os.path.join(frames_dir, f"frame_{frame_id:06d}.jpg")
                
                # Check if file exists (integrity)
                if os.path.exists(img_path):
                    session_data.append({
                        'img_path': img_path,
                        'action': action,
                        'frame_id': frame_id
                    })
            
            # Add to master list
            # We store (session_obj_ref, local_index)
            # No, cleaner: flat list of (img_path, action, *history_paths*)
            # Actually, generating history paths on the fly is better but requires knowing session boundaries.
            
            # Strategy: Samples list contains metadata strictly.
            # We need to know previous 3 frames for each sample. 
            # If index < 3, we pad.
            
            for i in range(len(session_data)):
                # Get current and previous k-1 frames
                frames_stack = []
                for k in range(self.stack_size):
                    prev_idx = i - k
                    if prev_idx >= 0:
                        frames_stack.append(session_data[prev_idx]['img_path'])
                    else:
                        # Padding: Use the oldest available frame (usually frame 0)
                        # or None (to indicate zero padding)
                        frames_stack.append(None) 
                
                # Frames are appended [current, prev, prev-1, prev-2]
                # We usually want [prev-2, prev-1, prev, current] order? 
                # VecFrameStack usually does channels: [Current, Prev, Prev-2...] or chronological?
                # SB3 VecFrameStack stacks on channel dim. 
                # Convention: Usually [Oldest ... Newest] or [Newest ... Oldest]?
                # SB3 NatureCNN implementation flattens channels.
                # Let's stick to Chronological: [T-3, T-2, T-1, T]
                
                frames_stack.reverse() # Now [T-3, T-2, T-1, T]
                
                self.samples.append({
                    'paths': frames_stack,
                    'action': session_data[i]['action']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_paths = sample['paths']
        action = sample['action']
        
        stacked_frames = []
        
        for p in image_paths:
            if p is None:
                # Padding: Zero frame (128x128)
                frame = np.zeros((128, 128), dtype=np.uint8)
            else:
                # Retrieve from Cache
                frame = self.image_cache.get(p)
                if frame is None:
                     # Should not happen if preloaded, but fallback
                     frame = np.zeros((128, 128), dtype=np.uint8)
            
            stacked_frames.append(frame)
            
        # Convert to Numpy Stack (4, 128, 128)
        # Note: Preprocessor returns (128, 128) squeezed.
        np_stack = np.array(stacked_frames, dtype=np.uint8)
        
        # Convert to Tensor
        # PyTorch expects Float 0-1 usually? 
        # SB3 "NatureCNN" handles normalization internally? 
        # Yes, SB3 CnnPolicy divides by 255 if "normalize_images=True".
        # But we are passing this DIRECTLY to the network feature extractor in our custom loop?
        # If we use SB3 policy, it expects input dict or tensor?
        # In custom loop, we'll feed tensor.
        # Let's return ByteTensor (0-255) and normalize in training loop to save dataloader bandwidth.
        
        state_tensor = torch.from_numpy(np_stack) # Shape (4, 128, 128)
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        return state_tensor, action_tensor
