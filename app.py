from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
from keras.models import load_model # type: ignore
from dotenv import load_dotenv
import time
from werkzeug.utils import secure_filename
import google.generativeai as genai
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model_name = "gemini-pro"
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 150,
}

# Allowed file check function
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your trained model
model = load_model('models/yakshagana.h5')

# Class names, mappings, and descriptions remain the same
class_names = ['Dataset(Yakshagana)', 'Bheemamudi', 'Devi mudi', 'Kesaritatti', 'Kirana', 
               'Kolukireeta', 'Kuttari', 'Mahisha', 'Pakdi Shaiva vaishnava', 'Turai']

vesha_mapping = {
    'Dataset(Yakshagana)': '0',
    'Bheemamudi': 'Rakshasa vesha',
    'Devi mudi': 'Devi',
    'Kesaritatti': 'Bannada Vesha',
    'Kirana': 'Strivesha',
    'Kolukireeta': 'Raja Vesha',
    'Kuttari': 'Hennu Banna',
    'Mahisha': 'Rakshasa vesha',
    'Pakdi Shaiva vaishnava': 'Pundu Vesha',
    'Turai': 'Pundu Vesha'
}

vesha_description = {
    'Dataset(Yakshagana)': '0',
    'Bheemamudi': 'This type of crown is especially used for roles like Bheema, Dushyasana, Ghatothkaj, Kumbhakarna, etc',
    'Devi mudi': 'This type of crown is used in characters like Devi, Vishnu, Bhoodevi, Laxmi, etc',
    'Kesaritatti': 'This type of crown is used for characters like Shumbha, Nishumbha, Raavana, Taarakasura, etc',
    'Kirana': 'Used for all female roles',
    'Kolukireeta': 'Used for roles like Arjuna, Devendra, Hiranyaksha, Rakthabeeja, etc',
    'Kuttari': 'Used for Shoorpankha, Taataki, Hidimbi, etc',
    'Mahisha': 'The Mahisha crown is typically worn by characters portraying Rakshasas (demons) in Yakshagana performances.',
    'Pakdi': 'This type of crown is used for young-aged characters like Abhimanyu, Agni, Varuna, Vayu, etc',
    'Turai': 'This type of crown is used for Prahlad, Lava, Kusha, Subrahmanya, Brahma, Vishnu, etc'
}

# Fallback descriptions if geminai won't work it'll display
fallback_descriptions = {
    'Dataset(Yakshagana)': 'A traditional Indian dance-drama from Karnataka, known for its elaborate costumes, makeup, and headgear.',
    'Bheemamudi': 'A majestic crown type used for powerful characters like Bheema. This headgear signifies strength and valor, typically worn by characters with robust personalities.',
    'Devi mudi': 'An ornate crown style used for divine feminine characters. This headpiece represents grace and divine authority, commonly worn by goddesses and celestial beings.',
    'Kesaritatti': 'A distinctive crown used for powerful demon characters. The design incorporates fierce elements to portray the character\'s intimidating nature.',
    'Kirana': 'An elegant headpiece designed specifically for female characters. It emphasizes feminine grace and beauty in the performance.',
    'Kolukireeta': 'A royal crown style used for noble warrior characters. This headgear symbolizes leadership and heroic qualities.',
    'Kuttari': 'A specialized headpiece for female demon characters. The design combines feminine elements with fierce characteristics.',
    'Mahisha': 'A dramatic crown style for demon king characters. The design emphasizes the powerful and fearsome nature of the character.',
    'Pakdi': 'A crown style for young divine or noble characters. The design represents purity and devotion.',
    'Turai': 'A distinctive crown used for young divine characters. The headpiece symbolizes divine wisdom and youth.'
}

# AI description generation function
def generate_ai_description(class_name):
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("API Key not found.")
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

        prompt = f"Provide a brief, single-paragraph description about the {class_name} character in Yakshagana, focusing on its specific usage as described: '{vesha_description.get(class_name, 'No description available.')}'. Explain why this character is used, its relationship with the {class_name} crown, and its significance in Yakshagana performances."

        # Request to Gemini API
        response = model.generate_content(prompt)

        # Check if the response contains a text field with a valid description
        if response.text and isinstance(response.text, str):
            ai_description = response.text.strip()
            if ai_description:
                return ai_description
            else:
                raise Exception("Empty response from Gemini API.")
        else:
            raise Exception(f"Gemini API returned an invalid or empty response.")

    except Exception as e:
        # Log the error details
        logger.error(f"Error generating description for {class_name}: {str(e)}")
        # Return the actual error message to the frontend
        return f"Error generating description: {str(e)}"




# API route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        image = Image.open(filepath)
        image = image.convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_tensor = np.expand_dims(image_array, axis=0)

        # Make prediction
        output = model.predict(image_tensor)
        predicted_class_index = np.argmax(output, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        predicted_vesha = vesha_mapping.get(predicted_class_name, 'Unknown')
        predicted_description = vesha_description.get(predicted_class_name, 'No description available.')

        # Generate AI description using Gemini
        ai_description = generate_ai_description(predicted_class_name)

        return jsonify({
            'predicted_class_name': predicted_class_name,
            'predicted_vesha': predicted_vesha,
            'predicted_description': predicted_description,
            'ai_description': ai_description,
            'image_filename': filename
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
