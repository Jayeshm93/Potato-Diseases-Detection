from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model('../saved_models/1')
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']



# def read_file_from_image(data):
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# @app.post('/potato-disease-predict')
# async def predict(
#         file: UploadFile = File(...)
# ):
#     image = read_file_from_image(await file.read())
#     img_batch = np.expand_dims(image, 0)  # expands 1 dims
#
#     # Detect disease
#     predictions= MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions)
#
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }


def read_file_from_image(data):
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize if model was trained on 0-1
    return image


@app.post('/potato-disease-predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_from_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # add batch dimension

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions)

    return {
        'class': predicted_class,
        'confidence': float(confidence * 100)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)