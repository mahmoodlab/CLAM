Install Conda Environment for CLAM and Classic MIL
===========

Create conda environment
```shell
conda env create -n simclr_reprod -f '/media/visiopharm5/WDGold/deeplearning/MIL/CLAM/finally2.yml'
```
It will fail with `CondaEnvException: pip failed` caused by `ERROR: Could not find a version that satisfies the requirement libtiff==0.5.0`.


Some pip dependencies need to be installed manually inside the created environment
```shell
conda activate simclr_reprod
pip install libtiff openslide_python pyvips gdal mapnik pyproj glymur javabridge -f https://girder.github.io/large_image_wheels
```


Navigate to a path suitable for cloning new repository and install `smooth-topk` (topk==1.0)
```shell
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```


Install the rest pip dependencies
```shell
pip install -r '/media/visiopharm5/WDGold/deeplearning/MIL/CLAM/finally2.txt'
```shell


If it fails with `Exception: Error finding javahome on linux: ['bash', '-c', 'java -version']`, and give you a blank output when typing `echo $JAVA_HOME`, you will need to configure java
```shell
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get install openjdk-11-jdk
```
Add `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64` to `~/.bashrc`. Verify `echo $JAVA_HOME` again to ensure that you have configured it correctly. And relaunch the pip installation
```shell
source ~/.bashrc # for current user
pip install -r '/media/visiopharm5/WDGold/deeplearning/MIL/CLAM/finally2.txt'
```


It should work this time!
