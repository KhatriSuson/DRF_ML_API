from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def upload_image(request):
    description = None
    image_url = None

    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        image_url = fs.url(filename)

        # Load and preprocess the image
        img_path = fs.path(filename)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Load pre-trained model (InceptionV3)
        model = InceptionV3(weights='imagenet')
        preds = model.predict(x)

        # Decode the results into a list of tuples (class, description, probability)
        decoded_preds = decode_predictions(preds, top=3)[0]
        description = ', '.join([f"{desc} ({prob:.2f})" for (_, desc, prob) in decoded_preds])

    return render(request, 'upload.html', {'description': description, 'image_url': image_url})

def test(request):
    return render(request, 'test.html')