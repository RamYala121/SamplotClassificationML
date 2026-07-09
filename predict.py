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


def predict_folder(net, folder_path, classes):
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))
             and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    if not files:
        print(f"No image files found in folder: {folder_path}")
        return

    for filename in sorted(files):
        image_path = os.path.join(folder_path, filename)
        label = predict_image(net, image_path, classes)
        print(f"{filename}: {label}")


def main():
    if len(sys.argv) == 1:
        home = os.path.expanduser("~")
        documents_folder = os.path.join(home, "Documents", "test_file")
        downloads_folder = os.path.join(home, "Downloads", "test_file")
        onedrive = os.environ.get("OneDrive")
        onedrive_documents_folder = None
        if onedrive:
            onedrive_documents_folder = os.path.join(onedrive, "Documents", "test_file")

        if os.path.exists(documents_folder):
            path = documents_folder
        elif onedrive_documents_folder and os.path.exists(onedrive_documents_folder):
            path = onedrive_documents_folder
        elif os.path.exists(downloads_folder):
            path = downloads_folder
        else:
            candidates = [documents_folder]
            if onedrive_documents_folder:
                candidates.append(onedrive_documents_folder)
            candidates.append(downloads_folder)
            print("No default folder found. Create one of these:")
            for candidate in candidates:
                print(f"  {candidate}")
            sys.exit(1)
        print(f"No path provided; using default folder: {path}")
    elif len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("Usage: python predict.py [path/to/image.jpg or path/to/folder]")
        sys.exit(1)

    if not os.path.exists(path):
        print(f"Path not found: {path}")
        sys.exit(1)

    model_path = os.path.join('logs', 'dog_vs_cat.pth')
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    net = load_model(model_path)
    classes = get_classes()

    if os.path.isdir(path):
        predict_folder(net, path, classes)
    else:
        label = predict_image(net, path, classes)
        print(f"Prediction: {label}")


if __name__ == '__main__':
    main()
