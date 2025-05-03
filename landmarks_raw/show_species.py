import argparse
import numpy as np
import os

import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Show the shape of landmarks for a given species')
    parser.add_argument('--species', type=str, help='Name of the species to visualize')
    parser.add_argument('--all', action='store_true', help='Show all species instead of a specific one')
    args = parser.parse_args()

    if args.all:
        # Get all npy files in the normalized directory
        normalized_dir = './normalized'
        all_files = [f for f in os.listdir(normalized_dir) if f.endswith('.npy')]
        
        if not all_files:
            print("No .npy files found in the normalized directory")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot each species with a different color
        colors = plt.cm.tab10.colors
        for i, file_name in enumerate(all_files):
            species_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(normalized_dir, file_name)
            
            try:
                landmarks = np.load(file_path)
                
                # Get mean shape if multiple samples
                if len(landmarks.shape) == 3:
                    shape = landmarks.mean(axis=0)
                else:
                    shape = landmarks
                
                color = colors[i % len(colors)]
                plt.scatter(shape[:, 0], shape[:, 1], label=species_name, color=color, s=30)
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        
        plt.title("All Species Landmarks")
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    else:
        # Check if species is provided when not using --all
        if not args.species:
            print("Error: Please specify a species with --species or use --all to show all species")
            return
            
        # Construct the path to the normalized npy file
        file_path = os.path.join('./normalized', f"{args.species}.npy")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
        
        # Load the data
        try:
            landmarks = np.load(file_path)
            print(f"Loaded landmark data with shape: {landmarks.shape}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
        
        # Visualize the landmarks
        plt.figure(figsize=(10, 8))
        
        # Assuming the data is in the format [n_samples, n_landmarks, 2]
        # We'll plot the mean shape
        if len(landmarks.shape) == 3:
            mean_shape = landmarks.mean(axis=0)
            plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c='blue')
            
            # Connect the landmarks with lines (optional)
            for i in range(len(mean_shape)):
                plt.annotate(str(i), (mean_shape[i, 0], mean_shape[i, 1]), 
                             fontsize=8, ha='center', va='center')
            
            plt.title(f"Mean landmark shape for {args.species}")
        else:
            # If it's a single shape [n_landmarks, 2]
            plt.scatter(landmarks[:, 0], landmarks[:, 1], c='blue')
            
            # Connect the landmarks with lines (optional)
            for i in range(len(landmarks)):
                plt.annotate(str(i), (landmarks[i, 0], landmarks[i, 1]), 
                             fontsize=8, ha='center', va='center')
            
            plt.title(f"Landmarks for {args.species}")
        
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

if __name__ == "__main__":
    main()