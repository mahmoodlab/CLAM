CLAM Installation Guide <img src="clam-logo.png" width="350px" align="right" />
===========
Next, use the environment configuration file to create a conda environment:
```shell
conda env create -f env.yml
```

Activate the environment:
```shell
conda activate clam_latest
```

If you want to use CONCH as the pretrained encoder, install the package in the environment by running the following command:
```shell
pip install git+https://github.com/Mahmoodlab/CONCH.git
```

When done running experiments, to deactivate the environment:
```shell
conda deactivate clam_latest
```
Please report any issues in the public forum.

[Return to main page.](README.md)
