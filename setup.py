# -*- coding: utf-8 -*-
from setuptools import setup, Command
import os, glob, shutil
from os.path import normpath, abspath, dirname, join
here = normpath(abspath(dirname(__file__)))
import versioneer

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

with open('doc_requirements.txt') as f:
    doc_reqs = f.read().splitlines()


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info ./__pycache__'.split(
        ' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(normpath(join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" %
                                     (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)


cmdclass = {'clean': CleanCommand}
cmdclass.update(versioneer.get_cmdclass())

setup(
    cmdclass=cmdclass,
    name='skultrafast',
    version=versioneer.get_version(),
    author='Till Stensitzki',
    author_email='tillsten@zedat.fu-berlin.de',
    url='http://github.com/tillsten/skultrafast',
    packages=['skultrafast', 'skultrafast.base_funcs'],
    package_data={'skultrafast': ['examples/data/test.npz', 'examples/data/messpyv1_data.npz', 'examples/data/germanium.npz']},
    license='LICENSE.txt',
    description='Python package for analyzing time-resolved spectra.',
    long_description=open('README.rst').read(),
    install_requires=install_reqs,
    keywords='science physics chemistry pump-probe spectroscopy time-resolved',
    python_requires='>=3.5',
    include_package_data=True,
)
