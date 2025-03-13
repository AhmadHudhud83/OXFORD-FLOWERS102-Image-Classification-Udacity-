# Predict.py File Implementation
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # forced using keras 2 instead of 3 to handle keras layers issue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import json
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import logging
from PIL import Image
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


img_size = 224
# Command Line app setup
parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
parser.add_argument("image_path",type=str,help="Image file path")
parser.add_argument("model_path",type=str,help="Model File path")
parser.add_argument("--top_k", type=int,default=1,help="Return top K classes")
parser.add_argument("--category_names",type=str,help="Path to JSON file for flower names")
args = parser.parse_args()

# Image procesisng function
def process_image(img):
  # Converting an image into a Tensor
  img = tf.convert_to_tensor(img,dtype=tf.float32)
  img = tf.image.resize(img, (img_size, img_size))
  print(img)

  # Normalization
  img = tf.cast(img, tf.float32)
  img /= 255.0

  # Converting back to numpy array
  final_img = img.numpy()
  return final_img

class_names = {}
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
#print("Class names here " , class_names)
model_path = args.model_path
loaded_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})

# prediction function definition
def predict(img_path,model,top_k):
  with Image.open(img_path) as im:
    # Image Processing before prediction
    img_converted = np.asarray(im)
    img_processed = process_image(img_converted)
    final_img = np.expand_dims(img_processed,axis=0)
    #print("Image Shape : ",final_img.shape)

    # Array of classes probs
    probs=model.predict(final_img)

    # Getting top K classes
    top_k_labels = np.argsort(probs[0])[-top_k:][::-1]
    # if optional category names file provided , then get the class names , else default value is the label itself
    top_k_class_names = [class_names.get(str(label), f"Class {label}") for label in top_k_labels]

    

    top_k_probs = probs[0][top_k_labels]

    return top_k_probs,top_k_class_names


test_img_path = args.image_path
probs,classes = predict(test_img_path,loaded_model,args.top_k)

with Image.open(test_img_path) as im:
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)

    ax1.imshow(im)
    ax1.axis('off')

    # Plot the class probabilities
    ax2.barh(range(args.top_k), probs, align='center')
    ax2.set_aspect(0.1)
    ax2.set_yticks(range(args.top_k))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()

print(probs)

# Thank you for Revewing my code !

