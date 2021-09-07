from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

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
    return JSONResponse({'hello': 'world'})


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

    img_class = model.predict_classes(img)  # returns ndim np_array
    img_class_index = img_class.item()  # extracting value(s)
    classname = class_names[img_class_index]

    # returns numpy array of class probabilities
    img_prob = model.predict_proba(img)
    data = [x * 100 for x in img_prob.tolist()[0]]
    first_five = sorted(data, reverse=True)[:5]
    classes = [class_names[data.index(x)].split(
        "__")[1].replace("_", " ") for x in first_five]

    res = {classes[i]: first_five[i] for i in range(len(classes))}

    pred_dict = {"class": classname, "probability": res}

    return JSONResponse(pred_dict)


async def list_disease(request):
    return class_names

async def cure(request):
    disease = request.query_params['disease']
    if not disease in class_names:
        return JSONResponse({"error": "not found"}, status_code=404)
    return disease

app = Starlette(debug=True, routes=[
    Route('/', homepage),
    Route('/predict', predict, methods=['POST']),
    Route('/diseases', list_disease, methods=['GET']),
    Route('/cure', cure, methods=['GET'])

])
