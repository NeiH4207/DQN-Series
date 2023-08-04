from setuptools import setup, find_packages

setup(
       name='DQN Series',
       version='0.1',
       packages=find_packages(),
       install_requires=[
           "matplotlib==3.7.1", 
           "numpy==1.24.3", 
           "scikit-learn==1.2.2", 
           "scipy==1.10.1", 
           "torch==2.0.1", 
           "gym==0.26.2",
           "pygame==2.5.0",
           "tqdm==4.65.0"
       ],
       entry_points={
           'console_scripts': [
           ],
       },
       license='MIT',
       description='A series of DQN, including DQN, Double DQN, Dueling DQN, Prioritized Experience Replay, and Rainbow.',
       long_description=open('README.md').read(),
       long_description_content_type='text/markdown',
       author='Vu Quoc Hien',
       author_email='neih4207@gmail.com',
       url='https://github.com/NeiH4207/DQN-Series',
   )
