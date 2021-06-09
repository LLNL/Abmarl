import fnmatch
import os


def custom_import_module(full_config_path):
    """
    Import and execute a python file as a module. Useful for import the experiment module and the
    analysis module.

    Args:
        full_config_path: Full path to the python file.

    Returns: The python file as a module
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mod", full_config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_from_trained_directory(full_trained_directory, checkpoint_desired):
    """
    Return the checkpoint directory to load the policy. If checkpoint_desired is specified and
    found, then return that policy. Otherwise, return the last policy.
    """
    checkpoint_dirs = find_dirs_in_dir('checkpoint*', full_trained_directory)

    # Try to load the desired checkpoint
    if checkpoint_desired is not None: # checkpoint specified
        for checkpoint in checkpoint_dirs:
            if checkpoint_desired == int(checkpoint.split('/')[-1].split('_')[-1]):
                return checkpoint, checkpoint_desired
        import warnings
        warnings.warn(
            f'Could not find checkpoint_{checkpoint_desired}. Attempting to load the last '
            'checkpoint.'
        )

    # Load the last checkpoint
    max_checkpoint = None
    max_checkpoint_value = 0
    for checkpoint in checkpoint_dirs:
        checkpoint_value = int(checkpoint.split('/')[-1].split('_')[-1])
        if checkpoint_value > max_checkpoint_value:
            max_checkpoint_value = checkpoint_value
            max_checkpoint = checkpoint

    if max_checkpoint is None:
        raise FileNotFoundError("Did not find a checkpoint file in the given directory.")

    return max_checkpoint, max_checkpoint_value


def find_dirs_in_dir(pattern, path):
    """
    Traverse the path looking for directories that match the pattern.

    Return: list of paths that match
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
