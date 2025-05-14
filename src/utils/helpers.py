import os
import sys
from pathlib import Path
import datetime

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
    return (num % den) == 0

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
