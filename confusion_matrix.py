############################################
# Confusion matrix script
# Evaluates the validation dataset and prints the confusion matrix
############################################

import os
import argparse

import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torchvision
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

from config import val_folder, transform_test, batch_size, num_workers
from net import Net, NetExample
from utils import make_pred_on_dataloader


def load_model(path, num_classes=2, in_channels=3):
    state = torch.load(path, map_location=torch.device('cpu'))
    try:
        net = Net(in_channels=in_channels, out_features=num_classes)
        net.load_state_dict(state)
        model_type = 'Net'
    except RuntimeError:
        net = NetExample(in_channels=in_channels, out_classes=num_classes)
        net.load_state_dict(state)
        model_type = 'NetExample'
    net.eval()
    return net, model_type


def build_val_dataloader(folder, batch_size, num_workers):
    dataset = torchvision.datasets.ImageFolder(root=folder, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return dataloader, dataset.classes, dataset.class_to_idx


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix from validation folders')
    parser.add_argument('--model', default=os.path.join('logs', 'dog_vs_cat.pth'),
                        help='Path to the trained model file')
    parser.add_argument('--data-folder', default=val_folder,
                        help='Path to the validation folder')
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help='Batch size for validation dataloader')
    parser.add_argument('--num-workers', type=int, default=num_workers,
                        help='Number of workers for dataloader')
    parser.add_argument('--output', default='confusion_matrix.png',
                        help='Output PNG file for the confusion matrix')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'Model file not found: {args.model}')
    if not os.path.isdir(args.data_folder):
        raise FileNotFoundError(f'Data folder not found: {args.data_folder}')

    net, model_type = load_model(args.model, num_classes=2, in_channels=3)
    val_loader, classes, class_to_idx = build_val_dataloader(args.data_folder, args.batch_size, args.num_workers)

    print(f'Using model checkpoint: {args.model}')
    print(f'Loaded model architecture: {model_type}')
    print(f'Data folder: {args.data_folder}')
    print(f'Data classes: {classes}')
    print(f'Batch size: {args.batch_size}')

    y_true, y_pred = make_pred_on_dataloader(net, val_loader)
    print('Predicted class counts:', Counter(y_pred))
    print('Actual class counts:', Counter(y_true))

    print('Predicted labels unique:', sorted(set(y_pred)))
    print('All predictions same:', len(set(y_pred)) == 1)

    sample_index = min(5, len(y_true))
    print(f'Sample predictions (first {sample_index}):')
    for i in range(sample_index):
        print(f'  Actual={classes[y_true[i]]}, Predicted={classes[y_pred[i]]}')

    sample_logits = get_sample_logits(net, val_loader, sample_index)
    print('Sample logits / probabilities:')
    for idx, (logit, prob) in enumerate(sample_logits):
        print(f'  Example {idx}: logit={logit.tolist()}, probs={prob.tolist()}')

    labels = [class_to_idx[c] for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    display_names = [c.title() for c in classes]
    print('\nClassification Report:\n')
    print(classification_report(y_true, y_pred, target_names=display_names, output_dict=False, zero_division=0))
    print('Confusion Matrix:')
    print(cm)
    print('Labels order:')
    print(display_names)

    plot_confusion_matrix(cm, display_names, args.output)


def get_sample_logits(net, dataloader, count=5):
    samples = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = net(images)
            probs = F.softmax(outputs, dim=1)
            for i in range(min(count - len(samples), outputs.size(0))):
                samples.append((outputs[i].cpu(), probs[i].cpu()))
            if len(samples) >= count:
                break
    return samples


def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Confusion matrix image saved to: {output_path}')


if __name__ == '__main__':
    main()
