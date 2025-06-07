import os

from typing import Optional
from pathlib import PureWindowsPath


def validate_file_path(
    file_path: str,
    must_exist: bool = True,
    allowed_extensions: Optional[list[str]] = None,
) -> str:

    # check that the fie_path is valid string
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("File path must be a non-empty string.")

    # make sure the file path is safe, block directory traversal
    if ".." in file_path or file_path.startswith("/"):
        raise ValueError("File path must not contain directory traversal characters.")

    # check if the file exists if must_exist is True
    if must_exist and not os.path.isfile(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if allowed_extensions is not None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in allowed_extensions:
            raise ValueError(
                f"File extension {ext} is not allowed. Allowed: {allowed_extensions}"
            )
    
    return PureWindowsPath(file_path).as_posix() if os.name == 'nt' else file_path
