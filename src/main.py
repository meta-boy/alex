import io
from datetime import datetime

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageDraw
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import matplotlib.pyplot as plt
import os
import shutil


templates = Jinja2Templates(directory='templates')

model_path = "./input/plant-diseases-classification-using-alexnet/AlexNetModel.hdf5"
model = load_model(model_path)
model.summary()


class_dict = {'Apple___Apple_scab': 0,
              'Apple___Black_rot': 1,
              'Apple___Cedar_apple_rust': 2,
              'Apple___healthy': 3,
              'Blueberry___healthy': 4,
              'Cherry_(including_sour)___Powdery_mildew': 5,
              'Cherry_(including_sour)___healthy': 6,
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
              'Corn_(maize)___Common_rust_': 8,
              'Corn_(maize)___Northern_Leaf_Blight': 9,
              'Corn_(maize)___healthy': 10,
              'Grape___Black_rot': 11,
              'Grape___Esca_(Black_Measles)': 12,
              'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
              'Grape___healthy': 14,
              'Orange___Haunglongbing_(Citrus_greening)': 15,
              'Peach___Bacterial_spot': 16,
              'Peach___healthy': 17,
              'Pepper,_bell___Bacterial_spot': 18,
              'Pepper,_bell___healthy': 19,
              'Potato___Early_blight': 20,
              'Potato___Late_blight': 21,
              'Potato___healthy': 22,
              'Raspberry___healthy': 23,
              'Soybean___healthy': 24,
              'Squash___Powdery_mildew': 25,
              'Strawberry___Leaf_scorch': 26,
              'Strawberry___healthy': 27,
              'Tomato___Bacterial_spot': 28,
              'Tomato___Early_blight': 29,
              'Tomato___Late_blight': 30,
              'Tomato___Leaf_Mold': 31,
              'Tomato___Septoria_leaf_spot': 32,
              'Tomato___Spider_mites Two-spotted_spider_mite': 33,
              'Tomato___Target_Spot': 34,
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
              'Tomato___Tomato_mosaic_virus': 36,
              'Tomato___healthy': 37}

class_names = list(class_dict.keys())


async def homepage(request):
    return templates.TemplateResponse('index.html', {'request': request})


async def predict(request):
    form = await request.form()
    filename = form["image"].filename
    contents = await form["image"].read()

    img = Image.open(io.BytesIO(contents))
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    img_class = model.predict_classes(img)
    print(img_class)  # returns ndim np_array
    img_class_index = img_class.item()  # extracting value(s)
    # returns numpy array of class probabilities
    img_prob = model.predict_proba(img)
    # pred_dict = {"Class": classname, "Probability": str(prediction_prob), "thing": img_prob.tolist()[0][img_class_index] }
    name = f"{datetime.now().timestamp()}.png"

    data = [x * 100 for x in img_prob.tolist()[0]]
    first_five = sorted(data, reverse=True)[:5]
    classes = [class_names[data.index(x)].split(
        "__")[1].replace("_", " ") for x in first_five]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(classes, first_five, width=0.4)
    print(classes, first_five)
    plt.xlabel("Diseases")
    plt.ylabel("Probabilities")
    plt.title("Plant Disease Classification Probability Chart")
    plt.xticks(rotation=30, wrap=True)
    plt.savefig(f'static/results/{name}')

    return templates.TemplateResponse('predict.html', {'request': request, 'image': name})


async def fetus_deletus():
    folder = 'static/results'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if not filename == ".gitignore":
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


app = Starlette(debug=True, routes=[
    Route('/', homepage),
    Route('/predict', predict, methods=['POST']),
    Mount('/static', StaticFiles(directory='static'), name='static')

], on_shutdown=[fetus_deletus])
