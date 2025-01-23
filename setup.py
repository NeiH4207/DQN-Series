from setuptools import setup, find_packages


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="DQN Series",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [],
    },
    license="MIT",
    description="A series of DQN, including DQN, Double DQN, Dueling DQN, Prioritized Experience Replay, and Rainbow.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vu Quoc Hien",
    author_email="neih4207@gmail.com",
    url="https://github.com/NeiH4207/DQN-Series",
)
