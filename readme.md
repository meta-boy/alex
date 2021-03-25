# Steps

1. Download the dataset

https://www.kaggle.com/vipoooool/new-plant-diseases-dataset

2. Clone this repo

```
git clone https://github.com/meta-boy/alex.git && cd alex
```

3. Make the input folder

```
mkdir input
```

4. Unzip the dataset in this folder

```
unzip ~/Downloads/archive.zip -d ./input/
```

5. Make the virtual environment

```
python3.8 -m pip install virtualenv
python3.8 -m venv env
source ./env/bin/activate
```

6. Install the dependencies

```
pip install -r requirements.txt
```

7. Build and train the model

```
python AlexNetModel.py
```

``NOTE: If there are missing dependencies then just install them via pip``