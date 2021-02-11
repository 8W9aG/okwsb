"""Setup the okwsb module."""
import os

from setuptools import setup

INSTALL_REQUIRES = []
with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"), "r"
) as requirments_txt_handle:
    INSTALL_REQUIRES = [
        x
        for x in requirments_txt_handle
        if not x.startswith(".") and not x.startswith("-e")
    ]

setup(
    name="okwsb",
    version="0.0.1",
    description="Stock Trading Bot based on Reinforcement Learning.",
    author="Will Sackfield",
    author_email="will.sackfield@gmail.com",
    packages=["okwsb"],
    install_requires=INSTALL_REQUIRES,
    entry_points={"console_scripts": ["okwsb=okwsb.main:main"]},
)
