import os


def check_existing_files(
    dir_path: str,
    filenames: list[str],
) -> list[str]:
    return [
        filename
        for filename in filenames
        if os.path.exists(os.path.join(dir_path, filename))
    ]
