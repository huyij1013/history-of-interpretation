# Boilerplate imports.
import tensorflow.compat.v1 as tf
import tf_slim as slim
import sys
import inception_v3
import saliency
import requests
import gradio as gr
import os
import wget
import tarfile

if not os.path.exists('inception_v3.ckpt'):
    wget.download("http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz")
    tar = tarfile.open("inception_v3_2016_08_28.tar.gz")
    tar.extractall()
    tar.close()
ckpt_file = './inception_v3.ckpt'
graph = tf.Graph()

with graph.as_default():
    images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        _, end_points = inception_v3.inception_v3(images, is_training=False, num_classes=1001)

        # Restore the checkpoint
        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)

    # Construct the scalar neuron tensor.
    logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]

    # Construct tensor for predictions.
    prediction = tf.argmax(logits, 1)

guided_backprop = saliency.GuidedBackprop(graph, sess, y, images)


def guided_vanilla(image):
    image = image / 127.5 - 1.0
    prediction_class = sess.run(prediction, feed_dict = {images: [image]})[0]
    vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(
    image, feed_dict = {neuron_selector: prediction_class})
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
    return vanilla_mask_grayscale.tolist()


# Download human-readable labels for ImageNet.
inception_net = tf.keras.applications.InceptionV3() # load the model

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def classify_image(inp):
    inp = inp.reshape((-1, 299, 299, 3))
    inp = tf.keras.applications.inception_v3.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


image = gr.inputs.Image(shape=(299, 299), label="Your Image")
label = gr.outputs.Label(num_top_classes=3, label="Classification")

examples = [["doberman.png"], ["cat.png"]]
title = "Interpretation Through Guided Back-Propagation"
description = "This is a demo of guided back-propagation, an interpretation method developed by Springenberg et al. The underlying model " \
              "is the Inception Net classifier. To " \
              "use the demo, simply upload an image (or click to load an example) and hit submit to see its predicted class. Then, hit the interpret button to " \
              "see an overlaid heatmap generated through guided back-propagation."
# article = """
# <p style='text-align: center'><a href='https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition'>A Visual History of Interpretation for Image Recognition</a></p>
# """

gr.Interface(classify_image, image, label, capture_session=True, interpretation=guided_vanilla, examples=examples,
             allow_flagging=False, analytics_enabled=False, title=title, description=description).launch()
