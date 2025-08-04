import os
import sys
import string
import random
from pathlib import Path
import datetime
import logging
import json

def first(it):
    return it[0]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def exists(x):
    return x is not None

def get_current_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_time

def cycle(dl):
    while True:
        for data in dl:
            yield data

def divisible_by(num, den):
    return num % den == 0

def is_odd(n):
    return not divisible_by(n, 2)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_debug():
    return True if sys.gettrace() else False


def get_file_list_with_extension(folder_path, ext):
    """
    Search for all files with the specified extension(s) in the given folder and its subfolders.

    Args:
    folder_path (str): Path to the folder where the search will be performed.
    ext (str or list of str): File extension(s) to search for, starting with a dot (e.g., '.ply').

    Returns:
    list: A list of file paths (in POSIX format) matching the specified extension(s).
    """
    files_with_extension = []

    # Ensure 'ext' is a list
    if isinstance(ext, str):
        ext = [ext]

    ext_set = {e.lower() for e in ext}

    folder_path = Path(folder_path)

    # Traverse all files recursively
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ext_set:
            files_with_extension.append(file_path.as_posix())
    
    return files_with_extension

def get_parent_directory(file_path):
    current_directory = os.path.dirname(file_path)
    parent_directory = os.path.dirname(current_directory)
    return parent_directory

def get_directory_path(file_path):
    return os.path.dirname(file_path)

def get_filename_wo_ext(file_path):
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]

def get_file_list(dir_path):
    file_path_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    file_path_list.sort()
    return file_path_list

def get_all_directories(root_path):
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            directories.append(os.path.join(dirpath, dirname))
    return directories

def filter_none_results(results):
    return [result for result in results if result is not None]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbose output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def get_or_create_file_list_json(dataset_dir_path, json_path, extension='.npz'):
    """
    Get or create a JSON file containing a list of files with a given extension under a directory.
    If the JSON file exists, load the file list from it; otherwise, generate the file list and save to JSON.

    Args:
        dataset_dir_path (str): Path to the dataset directory.
        json_path (str): Path to the JSON file to save/load the file list.
        extension (str): File extension to search for (default: '.npz').

    Returns:
        list: List of file paths with the specified extension.
    """

    if not os.path.exists(json_path):
        file_list = get_file_list_with_extension(dataset_dir_path, extension)
        with open(json_path, 'w') as f:
            json.dump(file_list, f)
    else:
        with open(json_path, 'r') as f:
            file_list = json.load(f)
    
    file_list.sort()
    
    return file_list

def generate_random_string(length, batch_size=1):
    chars = string.ascii_letters + string.digits
    pool = random.choices(chars, k=length * batch_size)

    return [''.join(pool[i*length:(i+1)*length]) for i in range(batch_size)]