############################################
# Predict a single image using the trained model
############################################

import os
import sys
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision

from config import transform_test, train_folder, CHANNELS, NUM_CLASSES
from net import Net


def load_model(path):
    state = torch.load(path, map_location=torch.device('cpu'))
    # Try loading into the main Net; if the checkpoint was created
    # from the smaller example network (`NetExample`) fall back to it.
    try:
        net = Net(in_channels=CHANNELS, out_features=NUM_CLASSES)
        net.load_state_dict(state)
    except RuntimeError:
        from net import NetExample
        net = NetExample(in_channels=CHANNELS, out_classes=NUM_CLASSES)
        net.load_state_dict(state)
    net.eval()
    return net


def get_classes():
    train_dataset = torchvision.datasets.ImageFolder(root=train_folder, transform=transform_test)
    return train_dataset.classes


def predict_image(net, image_path, classes):
    image = Image.open(image_path).convert('RGB')
    tensor = transform_test(image).unsqueeze(0)
    with torch.no_grad():
        outputs = net(tensor)
        _, predicted = torch.max(outputs, 1)
    label = classes[predicted.item()]
    return label


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    model_path = os.path.join('logs', 'dog_vs_cat.pth')
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    net = load_model(model_path)
    classes = get_classes()
    label = predict_image(net, image_path, classes)

    print(f"Prediction: {label}")


if __name__ == '__main__':
    main()
