from setuptools import setup, find_namespace_packages

setup(
    name="penquest-env",
    version="0.1.0",
    description="A reinforcement learning environment for the digital board game PenQuest.",
    url="https://www.pen.quest",
    author="Sebastian Eresheim, Alexander Piglmann, Simon Gmeiner, Thomas Petelin",
    author_email="sebastian.eresheim@fhstp.ac.at",
    license="",
    packages=find_namespace_packages(exclude=["build*", "dist*", "logs*"]),
    install_requires=[
        "asyncio>=3.4.3",
        "aioamqp>=0.15.0",
        "gymnasium>=0.28.1",
        "penquest-pkgs>=0.1.0",
    ],
    classifiers=[""]
)