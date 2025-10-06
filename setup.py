from setuptools import find_packages, setup

setup(
    name="omni_drones",
    author="zrz",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "plotly",
        "einops",
        "av", # for moviepy
        "pandas",
        # "multielo @ git+https://github.com/djcunningham0/multielo.git@v0.4.0",
        "h5py",
        "filterpy",
        "usd-core==23.2",
        "numpy==1.23.5",
        "urllib3==1.26.18",
        # "torchinfo",
        # "torchopt"
    ],
)
