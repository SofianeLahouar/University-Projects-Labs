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

ap = argparse.ArgumentParser()
ap.add_argument("-target", "--target", required=True,
                help="Select an image to do predication on")
args = vars(ap.parse_args())


# If the user is using pytorch with GPU, the GPU will be selected else the
# CPU will be used
if torch.cuda.is_available():
    def map_location(storage, loc): return storage.cuda()
else:
    map_location = 'cpu'

# A fonction to Load the model in the device (GPU or CPU)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.eval()


# Loading the facial mask classification model
loaded_model = load_checkpoint(
    "models/face_mask_classification/facial_mask_classification_model.pth")


# Loading the facial detection model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    "models/face_detection/deploy.prototxt.txt",
    "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")

# Load the test picture
image = cv2.imread("tests/" + args["target"])

# Save the height and width for later use
(h, w) = image.shape[:2]

# resize and normalize the picture for the face detection model
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))


# Pass the test picture through the model
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# A function to normalize the picture for the facial mask classification
normalize_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    if confidence > 0.2:
        # extract the 4 points that define the bounding box of the detected
        # face

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        #print(startX, startY, endX, endY)

        # Crop only the faces from the image
        cropped = image[startY:endY, startX:endX]

        # Create a copy of the image
        pil_image = Image.fromarray(cropped, mode="RGB")

        # Normalize the image
        pil_image = normalize_transforms(pil_image)

        # Repesent the image in one array
        image_model = pil_image.unsqueeze(0)

        # Pass the image through the model

        result = loaded_model(image_model)
        result.data = torch.exp(result.data * 100)
        print(result.data.numpy())
        arr = result.data.numpy()

        # Select the best predection with a max function
        _, maximum = torch.max(result.data, 1)
        prediction = maximum.item()

        # Output the confidance level of the model

        conf = result.data.tolist()[0][prediction] * 100
        print(conf)
        # To eliminate rare useless detection we set the confidance level to
        # 0.1
        if conf > 0.1:
            if prediction == 0:
                # Wearing a mask
                cv2.rectangle(image, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
            elif prediction == 1:
                # Not wearing a mask
                cv2.rectangle(image, (startX, startY),
                              (endX, endY), (0, 0, 205), 2)

# show the output image
cv2.imshow("Output", image)
#cv2.imshow("output rescaled", cv2.resize(image, (300, 300)) )
cv2.waitKey(0)
