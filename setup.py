from setuptools import setup

setup(
      name='genie',
      version='0.0.1',
      description='de novo protein design through equivariantly diffusing oriented residue clouds',
      packages=['genie'],
      install_requires=[
            'tqdm',
            'numpy',
            'torch',
            'scipy',
            'wandb',
            'tensorboard',
            'pytorch_lightning',
      ],
)