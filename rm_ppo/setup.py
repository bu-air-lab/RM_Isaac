from setuptools import setup, find_packages

setup(name='rm_ppo',
      version='1.0.2',
      author='David DeFazio',
      author_email='ddefazi1@binghamton.edu',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Extended rsl_rl to include state estimation',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
