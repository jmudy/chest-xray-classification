from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)

# Configuración de la aplicación
modelpath = "../checkpoint/mymodel.hdf5"
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model(filepath=modelpath)
target_size = (220, 220)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(file):
    img = image.load_img(file, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    return "Pneumonia" if result[0][0] == 1 else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = predict_image(filepath)

            return render_template('index.html', filename=filename, prediction=prediction)

        else:
            return render_template('index.html', error='Invalid file format')

    # Método GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
