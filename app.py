from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = load_model('./input')

with open("./input/label_num_to_disease_map.json") as f:
    class_map = json.load(f)

class_map2 ={int(k):v for k,v in class_map.items()}

def predict_cassava(img_path):
    TARGET_SIZE = 350
    test_image = Image.open(img_path)
    test_image = test_image.resize((TARGET_SIZE, TARGET_SIZE))
    test_image = np.expand_dims(test_image, axis = 0)
    predict = np.argmax(model.predict(test_image))

    return class_map2[predict]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "./static/" + img.filename	
		img.save(img_path)

		p = predict_cassava(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
