############################################
# Nicola Altini (2020)
#
# Run this script before training your CNN.
# It will split your train dataset in two folders:
#   - positive 
#   - negative
############################################

import os
from config import *

files_train_folder = os.listdir(train_folder)
files_train_folder = [file for file in files_train_folder if file.endswith('.jpg')]

classes = ['positive', 'negative']

for class_ in classes:
    class_folder = os.path.join(train_folder, class_)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

print("Len Train Folder Files = ", len(files_train_folder))

for file_train in files_train_folder:
    file_class = file_train[:3]
    current_path = os.path.join(train_folder, file_train)
    if file_class == 'pos':
        new_positive_path = os.path.join(train_folder, 'positive', file_train)
        os.rename(current_path, new_positive_path)
    elif file_class == 'neg':
        new_negative_path = os.path.join(train_folder, 'negative', file_train)
        os.rename(current_path, new_negative_path)
