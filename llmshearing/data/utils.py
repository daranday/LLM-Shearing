from pathlib import Path

import numpy as np


def get_npy_shape_and_dtype(filepath):
    with open(filepath, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    return shape, dtype


def calculate_npy_size(shape, dtype):
    return np.prod(shape)


def get_total_npy_size(directory):
    total_size = 0
    for npy_file in directory.rglob("*.npy"):
        shape, dtype = get_npy_shape_and_dtype(npy_file)
        total_size += calculate_npy_size(shape, dtype)
    return total_size


def main(root_directory):
    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"Error: {root_directory} is not a valid directory.")
        return

    for subdir in root_path.iterdir():
        if subdir.is_dir():
            total_size = get_total_npy_size(subdir)
            print(
                f"Total size of .npy files in {subdir.name}: {total_size / (1024**3):.2f} GB"
            )
