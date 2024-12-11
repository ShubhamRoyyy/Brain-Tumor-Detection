import numpy as np
import cv2
from keras.models import load_model
from PIL import Image


model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('C:\\Users\\SHUBHAM\\Desktop\\major project\\pred\\pred8.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = model.predict(input_img)

result_final=np.argmax(result,axis=1)
print("Predicted probabilities:", result)
print("Predicted class:", result_final[0])