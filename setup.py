from setuptools import setup, find_packages

setup(
    name='tampc',
    version='',
    packages=find_packages(),
    url='',
    license='',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='',
    test_suite='pytest',
    tests_require=[
        'pytest',
    ], install_requires=['pytest', 'gpytorch', 'matplotlib', 'numpy', 'scipy', 'scikit-learn', 'torch', 'tensorboardX',
                         'seaborn']
)
