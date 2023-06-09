import os


def list_files(folder="./", suffix="", recursive=False):
    """List all files.

    Parameters
    ----------
    suffix: filename must end with suffix if given, it can also be a tuple
    recursive: if recursive, return sub-paths
    """
    files = []
    if recursive:
        for path, _, fls in os.walk(folder):
            files += [os.path.join(path, f) for f in fls if f.endswith(suffix)]
    else:
        files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    return files