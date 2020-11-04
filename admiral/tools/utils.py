
def custom_import_module(full_config_path):
    """
    Import and execute a python file as a module. Useful for import the experiment module and the
    analysis module.

    Parameters:
        full_config_path: Full path to the python file.
    
    Returns: The python file as a module
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mod", full_config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_dirs_in_dir(pattern, path):
    """
    Traverse the path looking for directories that match the pattern.

    Return: list of paths that match
    """
    import os, fnmatch
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
