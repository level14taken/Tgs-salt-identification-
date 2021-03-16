#https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import io
import PIL
import matplotlib.pyplot as plt
import shutil
from model import get_model
from flask import Flask,jsonify,request,render_template
import time
import os

app = Flask(__name__)

model= get_model()

transform_test= torchvision.transforms.Compose([ 
torchvision.transforms.Pad((13,13,14,14), padding_mode='edge'),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=(0,0,0),std=(1,1,1,))])

@app.route('/', methods=['GET', 'POST'])
def hello_world():
 if request.method == 'GET':
  return render_template('index.html')
 if request.method == 'POST':
  print(request.files)
  data = request.files['file']
  pred,image=get_prediction(data.read())
  plt.figure(figsize=(20,10))
  plt.subplot(1,2,1)
  plt.imshow(image)
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.imshow(pred)
  new_graph_name = "graph" + str(time.time()) + ".png"

 for filename in os.listdir('static/'):
   if filename.startswith('graph'):  # not to remove other images
      os.remove('static/' + filename)
 plt.axis('off')
 plt.savefig('static/' + new_graph_name)

 return render_template("result.html", graph=new_graph_name)
 

def get_prediction(image_path):
    image1 = PIL.Image.open(io.BytesIO(image_path)).convert('RGB')
    image1= torchvision.transforms.Resize((101,101))(image1)
    image= transform_test(image1).unsqueeze(0)
    bin_,y_pred= model(image)
    y_pred = F.sigmoid(y_pred)*(bin_.view(-1,1,1,1)>0)
    y_pred= (y_pred>.5).squeeze(1).cpu().numpy()
    y_pred= (y_pred[:,13:-14,13:-14]).astype(np.uint8)
    y_pred= y_pred.reshape(101,101)
    
    return y_pred,np.array(image1)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,debug=True)
