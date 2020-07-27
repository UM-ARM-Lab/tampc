import subprocess
import os
from tampc import cfg
import logging

logger = logging.getLogger(__name__)


def main():
    model_dir = os.path.join(cfg.ROOT_DIR, 'models')
    model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    for f in model_files:
        # reject non-xacro
        if not f.endswith('.xacro'):
            continue

        if f.startswith('_'):
            continue

        name = os.path.splitext(f)[0]
        full_path = os.path.join(model_dir, f)

        command = "rosrun xacro xacro {} > {}.urdf".format(full_path, os.path.join(cfg.ROOT_DIR, name))
        logger.info(command)
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    main()
