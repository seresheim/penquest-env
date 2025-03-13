from setuptools import setup, find_namespace_packages

setup(
    name="penquest-env",
    version="0.2.1",
    description="A reinforcement learning environment for the digital board game PenQuest.",
    url="https://www.pen.quest",
    author="Sebastian Eresheim, Alexander Piglmann, Simon Gmeiner, Thomas Petelin",
    author_email="sebastian.eresheim@fhstp.ac.at",
    license="MIT License",
    packages=find_namespace_packages(exclude=["build*", "dist*", "logs*"]),
    install_requires=[
        "asyncio>=3.4.3",
        "gymnasium>=0.29",
        "websockets>=12.0",
        "penquest-pkgs>=0.2.0",
    ],
    classifiers=[""]
)