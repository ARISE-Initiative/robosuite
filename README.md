# robosuite

#### Installations

1. Clone the repository

```
git clone https://github.com/DaebangStn/robosuite.git
```

2. There is an error when repository name and package name are the same.

```
mv ./robosuite ./pkg
```

3. Install packages 

(setuptools has changed the way it handles editable installs
so that you should use `--config-settings editable_mode=compat` to install)

```
pip install -e . --config-settings editable_mode=compat
pip install -r requirements-extra.txt
```