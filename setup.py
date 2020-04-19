from setuptools import setup, find_packages

setup(name='kltpicker',
      version='0.1',
      description='KLT picker',
      url='http://github.com/dalitco54/kltpicker',
      author='Dalit Cohen',
      author_email='dalitcohen@mail.tau.ac.il',
      packages=find_packages(),
      license='MIT',
      install_requires=[
          'numpy',
          'mrcfile',
          'operator',
          'multiprocessing',
          'pathlib',
          'warnings',
          'sys',
          'argparse',
          'scipy',
          'matplotlib',
          'os',
          'pyfftw',
          'tqdm'
      ],
      python_requires='>=3',
      scripts=['bin/pick.py'],
      zip_safe=False)
