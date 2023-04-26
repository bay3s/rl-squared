import glob
import os


def cleanup_log_dir(log_dir: str) -> None:
    """
    Clean up logs.

    Args:
        log_dir (str): Directory in which logs have been stored.

    Returns:
        None
    """
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)
