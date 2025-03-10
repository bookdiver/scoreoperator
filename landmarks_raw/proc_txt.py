import os
import numpy as np
import pandas as pd

# get current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    metadata = pd.read_csv(os.path.join(CURR_DIR, "metadata.txt"), sep=";")
    aligned = pd.read_csv(os.path.join(CURR_DIR, "aligned_landmarks.txt"), sep=",", header=None)
    print(f"Read {len(metadata)} species from metadata.txt")
    
    if not os.path.exists(os.path.join(CURR_DIR, "speices")):
        os.makedirs(os.path.join(CURR_DIR, "speices"))
    
    print("The speices are:")
    for name in metadata["species"]:
        print(name)
        pts_flatten = np.array(aligned.loc[metadata["species"] == name])[0]
        pts = pts_flatten.reshape(-1, 2) * np.array([0.5/0.1, 0.55/0.1]) + np.array([0.5, 0.55])    # scale and shift to [0, 1]
        save_name = name.lower().replace(" ", "_") + ".npy"
        np.save(os.path.join(CURR_DIR, "speices", save_name), pts)
        
    print(f"Save all species to {os.path.join(CURR_DIR, 'speices')}")
    
if __name__ == "__main__":
    main()