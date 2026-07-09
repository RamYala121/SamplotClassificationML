############################################
# Simple GUI for Samplot classification
############################################

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from predict import load_model, get_classes, predict_image


def load_model_safe():
    model_path = os.path.join('logs', 'dog_vs_cat.pth')
    if not os.path.isfile(model_path):
        messagebox.showerror("Model not found", f"Model file not found:\n{model_path}")
        return None
    try:
        net = load_model(model_path)
        classes = get_classes()
    except Exception as err:
        messagebox.showerror("Load error", f"Failed to load model:\n{err}")
        return None
    return net, classes


def choose_image():
    filename = filedialog.askopenfilename(
        title="Select a samplot image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    if not filename:
        return
    selected_path.set(filename)
    result_var.set("")
    display_image(filename)
    predict_button.config(state=tk.NORMAL)


def display_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as err:
        messagebox.showerror("Image error", f"Unable to open image:\n{err}")
        return
    resample = getattr(Image, 'Resampling', None)
    if resample is not None:
        image.thumbnail((320, 320), resample.LANCZOS)
    else:
        image.thumbnail((320, 320), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


def do_predict():
    image_path = selected_path.get()
    if not image_path or not os.path.isfile(image_path):
        messagebox.showwarning("No image", "Please select an image first.")
        return
    try:
        label = predict_image(net, image_path, classes)
        result_var.set(label.capitalize())
    except Exception as err:
        messagebox.showerror("Prediction error", f"Failed to classify image:\n{err}")


def build_gui():
    root = tk.Tk()
    root.title("Samplot Classifier")
    root.geometry("520x600")
    root.resizable(False, False)

    global selected_path, result_var
    selected_path = tk.StringVar(root)
    result_var = tk.StringVar(root)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(frame, text="Samplot Classifier", font=("Arial", 18, "bold"))
    title_label.pack(pady=(0, 10))

    info_label = tk.Label(frame, text="Select a samplot image and click Predict.", font=("Arial", 11))
    info_label.pack(pady=(0, 10))

    choose_button = tk.Button(frame, text="Choose Image", command=choose_image, width=20)
    choose_button.pack(pady=(0, 10))

    path_label = tk.Label(frame, textvariable=selected_path, wraplength=480, justify=tk.LEFT)
    path_label.pack(pady=(0, 10))

    image_frame = tk.LabelFrame(frame, text="Preview", width=480, height=340)
    image_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
    image_frame.pack_propagate(False)

    global image_label
    image_label = tk.Label(image_frame)
    image_label.pack(expand=True)

    predict_btn = tk.Button(frame, text="Predict", command=do_predict, state=tk.DISABLED, width=20)
    predict_btn.pack(pady=(0, 10))

    global predict_button
    predict_button = predict_btn

    result_container = tk.Frame(frame)
    result_container.pack(fill=tk.X, pady=(10, 0))
    tk.Label(result_container, text="Result:", font=("Arial", 12)).pack(side=tk.LEFT)
    tk.Label(result_container, textvariable=result_var, font=("Arial", 12, "bold"), fg="#007700").pack(side=tk.LEFT, padx=(10, 0))

    root.mainloop()


if __name__ == '__main__':
    loaded = load_model_safe()
    if loaded is None:
        raise SystemExit(1)
    net, classes = loaded

    build_gui()
