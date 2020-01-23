import argparse
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(ns=None):
    parser = argparse.ArgumentParser(description="Remove directories holding empty runs")
    parser.add_argument("--runs", help="runs directory holding runs to clean up; default: %(default)s",
                        default="simulation/runs")
    parser.add_argument("--dry", help="dry run without removing anything", action="store_true")
    args = parser.parse_args(namespace=ns)
    return args


def main():
    args = parse_args()

    run_dirs = [d for d in os.listdir(args.runs) if os.path.isdir(os.path.join(args.runs, d))]
    # mark directories to be removed
    remove_dirs = []
    for d in run_dirs:
        # see if this directory is empty
        full_path = os.path.join(args.runs, d)
        root_dir = Path(full_path)
        dir_size = sum(f.stat().st_size for f in root_dir.glob('**/*') if f.is_file())
        if dir_size is 0:
            remove_dirs.append(full_path)
        logger.info("%d %s", dir_size, full_path)

    if not args.dry:
        for d in remove_dirs:
            shutil.rmtree(d)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    main()
