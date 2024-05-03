from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np
from PIL import Image


app = Flask(__name__)

# Load the trained model
model = load_model('SnakeIdentifyModel.h5')

def preprocess_image(file):
    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the API endpoint for predicting the snake species
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_array = preprocess_image(file)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    snake_classes = ['cobra', 'common_krait', 'Green Vine Snake','King Cobra','ratsnake','Rock Python','Russells Viper','Saw Scaled Viper',
                     'Trinket Snake', 'wolf snake', 'trinketsnake'] 
    predicted_snake = snake_classes[predicted_class]

    return jsonify({'prediction': predicted_snake})

if __name__ == '__main__':
    app.run(debug=True)
