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

8. Move the trained models

```
mv *.hdf5 ./input/plant-diseases-classification-using-alexnet
```
9. Run the server

```
python app.py
```

8. Predict!

```
curl --request POST \
  --url http://127.0.0.1:5000/predict \
  --header 'Content-Type: multipart/form-data; boundary=---011000010111000001101001' \
  --form image=@/path/to/the/goddamn/image.png
```

``NOTE: If there are missing dependencies then just install them via pip``