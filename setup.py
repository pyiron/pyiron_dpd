"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name='pyiron_dpd',
    version=versioneer.get_version(),
    description='pyiron_dpd - module extension to pyiron.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_dpd',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='@mpie.de',
    license='BSD',

    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10'],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'pyiron_base==0.6.12',
        'pyiron_atomistic==0.2.67',
        'pyiron_contrib==0.1.10',
        'maxvolpy==0.3.8',
        'pyiron_data==0.0.22',
        'seaborn>=0.12',
        'scikit-learn>=1',
        'tqdm',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    cmdclass=versioneer.get_cmdclass(),

    )
