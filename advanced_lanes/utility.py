from pathlib import Path


def get_output_folder_path():
    return Path(__file__).resolve().parents[1].joinpath("output_images")

def get_input_dir_path(dir):
    return Path(__file__).resolve().parents[1].joinpath(dir)
