from setuptools import setup,find_packages

setup(name='Tetris',
      version='0.0.1',
      install_requires=['gym', 'numpy'],#And any other dependencies required
      packages=find_packages()
)