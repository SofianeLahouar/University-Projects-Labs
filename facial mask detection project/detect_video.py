import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-target", "--target", required=True,
                help="The URL of the camera output feed")
args = vars(ap.parse_args())

# fonction eli ta3mel el transformation mta3 el tsawer
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# fonction tloadi beha el model b pytorch


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.eval()


# Hedha el model eli 5demenh bel kaggle fichier 200 mb taw nhawel nsobha
# el lila fel drive
loaded_model = load_checkpoint(
    "models/face_mask_classification/facial_mask_classification_model.pth")


# Hedha model eli ya3mel detection mta3 el wejouh (mawjoud 3al internet)
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    "models/face_detection/deploy.prototxt.txt",
    "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")

#url = 'http://192.168.1.3:8080/video'
url = args["target"]
cap = cv2.VideoCapture(url)
while(True):
    ret, image = cap.read()
    if image is not None:
        # ne5dhou fel height wel width mta3 el taswira
        (h, w) = image.shape[:2]
        # houni nresiziw fel taswira el 300x300
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # houni 3adina el taswira 3al face detector
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            #f = 1
            # Houni confidence mahtouta low 5ater sa3et inajem ikoun fama wejeh
            # ama el confidence a9al men 50%
            if confidence > 0.8:
                # Houni 5arjna el corr mta3 el 4 points eli ya3mlou el wejeh
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # f=f+1
                # houni croppina el wejeh wahdou
                print(startX, startY, endX, endY)
                cropped = image[startY:endY, startX:endX]
                #cv2.imshow("Output"+ str(f), cropped)
                # cv2.waitKey(0)
                # Juste 3mal copie mel image
                pil_image = Image.fromarray(cropped, mode="RGB")
                # hedhi el fonction mta3 el transfrom eli defineha el fou9
                pil_image = train_transforms(pil_image)
                # houni unsquezze sta3melneha bech nrodou el taswira
                # represented fi array wehed barka
                image_model = pil_image.unsqueeze(0)
                # 3andina el taswira 3al model
                result = loaded_model(image_model)
                result.data = torch.exp(result.data * 100)
                print(result.data.numpy())
                arr = result.data.numpy()
                # houni el resultat bech ta3tina 2 output, el akber howa el
                # a9rab lel predection eli 3malha el model

                _, maximum = torch.max(result.data, 1)
                prediction = maximum.item()
                # ken el output el lowel howa akber donc lebes mask, ken el
                # output el theny akber donc mahouch lebes mask
                conf = result.data.tolist()[0][prediction] * 100
                print(conf)
                if conf > 1:
                    if prediction == 0:
                        cv2.rectangle(
                            image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    elif prediction == 1:
                        cv2.rectangle(
                            image, (startX, startY), (endX, endY), (0, 0, 205), 2)
    cv2.imshow('frame', image)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()
