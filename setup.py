from setuptools import setup
from ccma import __version__

setup(
    name='ccma',
    version=__version__,
    packages=['ccma'],
    url='https://github.com/UniBwTAS/ccma',
    author='Thomas Steinecker',
    author_email='thomas.steinecker@unibw.de',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    description="Curvature Corrected Moving Average: An accurate and model-free path smoothing algorithm.",
)