from setuptools import setup

setup(name='kltpicker', 
      version='0.1',
      description='KLT picker',
      url='http://github.com/dalitco54/kltpicker',
      author='Dalit Cohen',
      author_email='dalitcohen@mail.tau.ac.il',
      packages=['kltpicker'],
      install_requires=[
          'numpy',
          'mrcfile',
          'operator',
          'glob',
          'argparse',
          'scipy',
          'matplotlib',
          'os',
          'pyfftw'
      ],
      zip_safe=False)
