import numpy as np
import tensorflow
from PIL import Image, ImageOps

def classification(img, weights_file):

    # Load the model
    model = tensorflow.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize and normalize the image
    image = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    normalized_image_array = np.array(image, dtype=np.float32) / 255

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the inference
    pred = np.argmax(model.predict(data)[0])

    return pred # Return position of the highest probability
