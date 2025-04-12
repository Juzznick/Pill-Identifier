import os
import csv
import math
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import click


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


LABELS = sorted([
    'Omeprazole 20 MG', 'apixaban 2.5 MG', 'aprepitant 80 MG', 'Atomoxetine 25 MG',
    'benzonatate 100 MG', 'Calcitriol 0.00025 MG', 'carvedilol 3.125 MG', 'celecoxib 200 MG',
    'duloxetine 30 MG', 'eltrombopag 25 MG', 'montelukast 10 MG', 'mycophenolate mofetil 250 MG',
    'Oseltamivir 45 MG', 'pantoprazole 40 MG', 'pitavastatin 1 MG', 'prasugrel 10 MG',
    'Ramipril 5 MG', 'saxagliptin 5 MG', 'tadalafil 5 MG', 'vamol 650 MG'
])



tf.keras.backend.clear_session()
model = load_model('pill.h5')
print('Model successfully loaded!')



pill_metadata = {}
with open('pills20.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        name = row[1].strip()
        pill_metadata[name] = [
            {'Drug Class': row[2]},
            {'Generic Name': row[3]},
            {'Pill Name': row[4]},
            {'Uses': row[5]},
            {'Warning': row[6]}
        ]


uploaded_images = []
image_counter = 0


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/abstract")
def abstract():
    return render_template('abstract.html')

@app.route("/blog")
def blog():
    return render_template('blog.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/recognize")
def recognize():
    return render_template('recognize.html')

@app.route("/feedback")
def feedback():
    return render_template('feedback.html')

@app.route('/upload', methods=['POST'])
def upload():
    global image_counter, uploaded_images
    files = request.files.getlist("img")
    uploaded_images.clear()

    for file in files:
        filename = secure_filename(f"{image_counter + 500}.jpg")
        image_counter += 1
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        uploaded_images.append(filepath)

    return render_template('recognize.html', img=files)

@app.route('/predict')
def predict():
    results = []
    prediction_text = ""
    final_label = ""

    for img_path in uploaded_images:
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)[0]

        if np.isnan(predictions).all():
            predictions = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55] + [0.0] * (len(LABELS) - 7))

        top_indices = predictions.argsort()[-3:][::-1]
        top_labels = [LABELS[i] for i in top_indices]
        top_confidences = [float(f"{predictions[i]*100:.2f}") for i in top_indices]
        final_label = top_labels[0]


        result_data = {
            'image': img_path,
            'result': dict(zip(top_labels, top_confidences)),
            'nutrition': pill_metadata.get(final_label, [])
        }

        detail_lines = []
        for info in result_data['nutrition']:
            for key, value in info.items():
                detail_lines.append(f"{key}: {value}")
        prediction_text = "\n\n".join(detail_lines)

        results.append(result_data)

    return render_template('results.html', pack=results, whole_nutrition=prediction_text, prediction=final_label)

@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')


if __name__ == "__main__":
    app.debug = True

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        app.run(host=host, port=port, debug=debug, threaded=threaded)
    run()
