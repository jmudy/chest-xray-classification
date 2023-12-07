import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

modelpath = "../checkpoint/mymodel.hdf5"
model = load_model(filepath=modelpath)

target_size = (220, 220)

img = image.load_img(
    "../img_tests/pneumonia.jpeg",
    target_size=target_size
    )
plt.imshow(img)

img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

result = model.predict(img)
print(result)

if result[0][0] == 1:
    prediction = "Pneumonia"
else:
    prediction = "Normal"

print("El paciente tiene {}".format(prediction))