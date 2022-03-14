import json
from io import BytesIO
import time
import os
import base64
import boto3
import time
import numpy as np
from PIL import Image
from skimage import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

import torchvision
from torchvision import transforms  # , utils
from network import model
from network import utils

def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def predict(net, item):
    
    sample = preprocess(item)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(sample["image"].unsqueeze(0).float())
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = Image.fromarray(predict_np * 255).convert("RGB")

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img


def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models(s3, bucket):
    net=model.U2NET(3,1)
    response = s3.get_object(Bucket=bucket, Key=f"models/u2net/u2net.pth")
    state = torch.load(BytesIO(response["Body"].read()),map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    return net


s3 = boto3.client("s3")
bucket = "sagemaker-m-model"
    
model = load_models(s3, bucket)
print(f"models loaded ...")

def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    image = image.convert("RGB")
    output = predict(model, np.array(image))
    output = output.resize((image.size), resample=Image.BILINEAR) # remove resample
    empty_img = Image.new("RGBA", (image.size), 0)
    new_img = Image.composite(image, empty_img, output.convert("L"))
    result = {"output": img_to_base64_str(new_img)}

    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json",
        },
    }
    
    