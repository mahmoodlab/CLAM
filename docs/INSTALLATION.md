CLAM Installation Guide <img src="clam-logo.png" width="350px" align="right" />
===========
For instructions on installing anaconda on your machine (download the distribution that comes with python 3):
https://www.anaconda.com/distribution/

After setting up anaconda, first install openslide:
```shell
sudo apt-get install openslide-tools
```

Next, use the environment configuration file located in **docs/clam.yaml** to create a conda environment:
```shell
conda env create -n clam -f docs/clam.yaml
```

Activate the environment:
```shell
conda activate clam
```

Once inside the created environment, to install smooth-topk (first cd to a location that is outside the project folder and is suitable for cloning new git repositories):

```shell
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```

When done running experiments, to deactivate the environment:
```shell
conda deactivate clam
```
Please report any issues in the public forum.

[Return to main page.](README.md)
