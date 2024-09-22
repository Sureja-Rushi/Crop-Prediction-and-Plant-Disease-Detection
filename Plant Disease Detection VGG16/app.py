from flask import Flask, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Home route
@app.route('/', methods=['POST'])
def home():
    # Check if an image file is part of the request
    if 'image' not in request.files:
        return "No image file in the request", 400
    
    # Get the image file from the request
    file = request.files['image']
    
    # Open the image file using PIL
    image = Image.open(file)

    # Resize the image to 400x400
    image = image.resize((400, 400))

    # Convert the image to a NumPy array
    image = np.array(image)

    model = load_model('cnn_model3.h5')

    if image.shape[-1] != 3:
        image = np.stack([image] * 3, axis=-1)  # In case the image is grayscale

# Normalize the image (scale pixel values to the range [0, 1])
    image = image / 255.0

# Add batch dimension (1, 400, 400, 3)
    image = np.expand_dims(image, axis=0)

# Make predictions
    predictions = model.predict(image)

    categories = ['Cashew Anthracnose', 'Cashew gumosis', 'Cashew Healthy', 'Cashew Leaf Miner', 'Cashew Red Rust', 'Cassava Bacterial Blight', 'Cassava Brown Spot', 'Cassava Green Mite', 'Cassava Healthy', 'Cassava Mosaic Disease', 'Maize Fall armyWorm', 'Maize grasshoper', 'Maize Healthy', 'Maize Leaf beetle', 'Maize Leaf Blight', 'Maize Leaf Spot', 'Maize Streak Virus', 'Tomato Healthy', 'Tomato Leaf Blight', 'Tomato Leaf Curl', 'Tomato Septoria Leaf Spot', 'Tomato Verticulium wilt']

    return categories[np.argmax(predictions)]

if __name__ == '__main__':
    app.run(debug=True)
