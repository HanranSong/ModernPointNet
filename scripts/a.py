import numpy as np

# Path to your .npy file
file_path = "./data/test_dataset_medium_crowd/gt/frame0.npy"

# Load the array
arr = np.load(file_path)

# Print the whole array (if it’s small)
print(arr)

# Or, if it’s large, print its shape and dtype, plus the first few entries:
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("First 10 elements:", arr.ravel()[:10])
