import os

from .. import HPG_BASE_DIR
from .types import PathType


def get_path(path_type: PathType, base_dir: str = HPG_BASE_DIR, 
             makedir: bool = False, require_exists: bool = False) -> str:
    """
    Returns the path associated with the path type. Pass `makedir` to create the
    folder. Pass `require_exists` to raise an error if the directory does not exist.

    The `path_type` must match a supported `PathType`.
    """

    # --- Build Path ---
    if path_type == "raw":
        path: str = os.path.join(base_dir, "physionet.org", "files", 
                                 "mimic-iv-demo-omop", "0.9", "1_omop_data_csv")

    elif path_type in ("cleaned", "metadata", "summary"):
        # -- Metadata and summaries sit inside the cleaned directory --
        cleaned_dir: str = os.path.join(base_dir, "cleaned_data")

        if path_type == "metadata":
            path = os.path.join(cleaned_dir, "metadata")
        elif path_type == "summary":
            path = os.path.join(cleaned_dir, "summary")
        else:
            path = cleaned_dir

    else:
        raise ValueError(f"Invalid path_type: {path_type}")

    # --- Optionally Make the Folder ---
    if makedir:
        os.makedirs(path, exist_ok=True)

    # --- Raise an Error if Not Exists Optionally ---
    if require_exists and not os.path.exists(path):
        raise AssertionError(f"Expected path does not exist: '{path}'.")

    return path
