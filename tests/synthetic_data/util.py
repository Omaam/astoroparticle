"""Utility module.
"""
import os


def join_and_create_directory(a, *paths, exist_ok=True):
    file_path = os.path.join(a, *paths)
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=exist_ok)
    return file_path
