FROM tensorflow/tensorflow:2.4.3
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt --no-cache-dir

RUN mkdir -p ./input/plant-diseases-classification-using-alexnet

RUN gdown 'https://drive.google.com/u/1/uc?id=1h2ImTgaH5TDcUUZd-KXbZFZNq98aMpZN'
RUN gdown 'https://drive.google.com/u/1/uc?id=1pbYW0-b7j2QAWedsU3sTi76-cW4aEABq'
RUN mv ./AlexNetModel.hdf5 ./input/plant-diseases-classification-using-alexnet/
RUN mv ./best_weights_9.hdf5 ./input/plant-diseases-classification-using-alexnet/



ENTRYPOINT ["python3", "app.py"]