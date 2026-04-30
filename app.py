import gradio as gr
import tensorflow as tf
import numpy as np
import os
import zipfile

# Unzip and load model
if not os.path.exists("saved_model.pb"):
    print("Unzipping model...")
    with zipfile.ZipFile("pneumonia_savedmodel.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    print("✅ Unzipped!")

model = tf.saved_model.load(".")
infer = model.signatures["serving_default"]

def is_likely_xray(img_array):
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
    rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))
    avg_diff = (rg_diff + rb_diff + gb_diff) / 3
    mean_brightness = np.mean(img_array)
    if avg_diff > 20:
        return False, "Image appears to be colorful — not a chest X-ray"
    if mean_brightness > 200:
        return False, "Image appears too bright — not a chest X-ray"
    return True, "OK"

def predict(image):
    if image is None:
        return "⚠️ Please upload an image"
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    valid, reason = is_likely_xray(img_array)
    if not valid:
        return f"⚠️ INVALID IMAGE\n{reason}\nPlease upload a real chest X-ray image."
    img_normalized = img_array / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    tensor = tf.constant(img_input)
    result = infer(input_layer_1=tensor)
    prediction = list(result.values())[0].numpy()[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    if confidence < 0.75:
        return f"⚠️ UNCERTAIN\nThe model is not confident enough ({confidence*100:.1f}%)\nPlease upload a clearer chest X-ray image."
    if prediction > 0.5:
        label = "🔴 PNEUMONIA DETECTED"
        conf_display = prediction * 100
    else:
        label = "🟢 NORMAL"
        conf_display = (1 - prediction) * 100
    return f"{label}\nConfidence: {conf_display:.1f}%"

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Text(label="Result"),
    title="🫁 Pneumonia Detector",
    description="Upload a chest X-ray image to check for pneumonia. Only grayscale chest X-rays will be analyzed."
)

app.launch()
