import os
from glob import glob
import shutil


def create_associations():
    rgb_dir = "/home/ak/GuidedResearch/data/rgbd_dataset_freiburg3_walking_halfsphere/rgb.txt"
    depth_path = "/home/ak/GuidedResearch/data/rgbd_dataset_freiburg3_walking_halfsphere/depth.txt"
    output_path = "/home/ak/GuidedResearch/data/rgbd_dataset_freiburg3_walking_halfsphere.txt"
    
   
    
    # Get depth entries
    depth_entries = []
    with open(depth_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            depth_data = line.strip().split(' ')
            depth_entries.append((float(depth_data[0]), depth_data[1]))
            
    # Get rgb entries
    rgb_entries = []
    with open(rgb_dir, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            rgb_data = line.strip().split(' ')
            rgb_entries.append((float(rgb_data[0]), rgb_data[1]))
   
    
    # Sort entries
    rgb_entries.sort(key=lambda x: x[0])
    depth_entries.sort(key=lambda x: x[0])
    
    # Write associations
    with open(output_path, 'w') as out_file:
        for i in range(min(len(rgb_entries), len(depth_entries))):
            out_file.write(f"{rgb_entries[i][0]:.6f} {rgb_entries[i][1]} {depth_entries[i][0]:.6f} {depth_entries[i][1]}\n")

if __name__ == "__main__":
    create_associations()