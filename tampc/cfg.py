import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')

PIC_SAVE_DIR = os.path.expanduser('~/Downloads/temp/pics')


def _file_name(category):
    def filename(experiment, subcategory=None):
        name = category
        if subcategory is not None:
            name = "{}_{}".format(name, subcategory)
        if type(experiment) is str:
            return os.path.join(DATA_DIR, '{}/{}.mat'.format(str(experiment), name))
        else:
            return os.path.join(DATA_DIR, 'epsilon/{}_{}.mat'.format(str(experiment), name))

    return filename


data_file = _file_name('data')
