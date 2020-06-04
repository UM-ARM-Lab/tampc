from setuptools import setup, find_packages

setup(
    name='meta_contact',
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
    ], install_requires=['pytest', 'gpytorch', 'matplotlib', 'numpy', 'scikit-learn', 'torch', 'tensorboardX', 'seaborn']
)
