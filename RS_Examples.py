#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 17:06
# @Author  : huyijiang
# @Email   : 844715480@qq.com
# @File    : RS_Examples.py
# @Software: PyCharm


# Boilerplate imports.
import tensorflow.compat.v1 as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import tf_slim as slim
import inception_v3
import saliency
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import config as cfg
import resnet
import cv2 as cv

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    # im = ((im + 1) * 127.5).astype(np.uint8)
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def ShowDivergingImage(grad, title='', percentile=99, ax=None):
    if ax is None:
        fig, ax = P.subplots()
    else:
        fig = ax.figure

    P.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    return im / 127.5 - 1.0


graph = tf.Graph()

with graph.as_default():


    input_img = tf.placeholder(tf.float32,[None, cfg.image_height, cfg.image_width, 3])
    cls_score = resnet.resnet_base(input_img, scope_name=cfg.NET_NAME, is_training=False)
    init = tf.global_variables_initializer()
    output_dir = r'F:\learning\interpretation\Tensorflow-Resnet-Image-Classification-master\output_2'
    ckpt_file = tf.train.latest_checkpoint(output_dir)
    neuron_selector = tf.placeholder(tf.int32)
    y = cls_score[0][neuron_selector]
    prediction = tf.argmax(cls_score, 1)

    # with tf.Session(graph=graph) as sess:

    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    variables = tf.global_variables()
    saver = tf.train.Saver(variables)
    init = tf.global_variables_initializer()

    sess.run(init)
    saver.restore(sess, ckpt_file)



imagepath = r'F:\learning\interpretation\Tensorflow-Resnet-Image-Classification-master\UCMerced_LandUse\1\train\airplane\airplane35.tif'
test_image_single_1 = cv.imdecode(np.fromfile(imagepath, dtype=np.uint8), 1)
test_image_single = cv.resize(test_image_single_1, (cfg.image_height, cfg.image_width))
test_image_single = np.reshape(test_image_single, [1, cfg.image_height, cfg.image_width, 3])
im = np.reshape(test_image_single_1, [cfg.image_height, cfg.image_width, 3])
test_image_single.astype(np.float32)
cls_score_1 = sess.run(cls_score, feed_dict={input_img: test_image_single})

print(cls_score_1)




# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
gradient_saliency = saliency.GradientSaliency(graph, sess, y, input_img)

# Compute the vanilla mask and the smoothed mask.
vanilla_mask_3d = gradient_saliency.GetMask(test_image_single[0], feed_dict = {neuron_selector: 1})
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(test_image_single[0], feed_dict = {neuron_selector: 1})

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))



# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
# NOTE: GuidedBackprop creates a copy of the given graph to override the gradient.
# Don't construct too many of these!
guided_backprop = saliency.GuidedBackprop(graph, sess, y, input_img)

# Compute the vanilla mask and the smoothed mask.
vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(test_image_single[0], feed_dict = {neuron_selector: 1})
smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(test_image_single[0], feed_dict = {neuron_selector: 1})

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Guided Backprop', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad Guided Backprop', ax=P.subplot(ROWS, COLS, 2))



# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
xrai_object = saliency.XRAI(graph, sess, y, input_img)

# Compute XRAI attributions with default parameters
xrai_attributions = xrai_object.GetMask(test_image_single[0], feed_dict={neuron_selector: 1})

# Set up matplot lib figures.
ROWS = 1
COLS = 3
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Show original image
ShowImage(im, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

# Show XRAI heatmap attributions
ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

# Show most salient 30% of the image
mask = xrai_attributions > np.percentile(xrai_attributions, 70)
im_mask = np.array(im)
im_mask[~mask] = 0
ShowImage(im_mask, title='Top 30%', ax=P.subplot(ROWS, COLS, 3))



# Create XRAIParameters and set the algorithm to fast mode which will produce an approximate result.
xrai_params = saliency.XRAIParameters()
xrai_params.algorithm = 'fast'

# Compute XRAI attributions with fast algorithm
xrai_attributions_fast = xrai_object.GetMask(test_image_single[0], feed_dict={neuron_selector: 1}, extra_parameters=xrai_params)

# Set up matplot lib figures.
ROWS = 1
COLS = 3
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Show original image
ShowImage(test_image_single[0], title='Original Image', ax=P.subplot(ROWS, COLS, 1))

# Show XRAI heatmap attributions
ShowHeatMap(xrai_attributions_fast, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

# Show most salient 30% of the image
mask = xrai_attributions_fast > np.percentile(xrai_attributions_fast, 70)
im_mask = np.array(test_image_single[0])
im_mask[~mask] = 0
ShowImage(im_mask, 'Top 30%', ax=P.subplot(ROWS, COLS, 3))




# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
integrated_gradients = saliency.IntegratedGradients(graph, sess, y, input_img)
blur_ig = saliency.BlurIG(graph, sess, y, input_img)

# Baseline is a black image for vanilla integrated gradients.
baseline = np.zeros(test_image_single[0].shape)
baseline.fill(-1)

# Compute the vanilla mask and the Blur IG mask.
vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
  test_image_single[0], feed_dict = {neuron_selector: 1}, x_steps=25, x_baseline=baseline)
blur_ig_mask_3d = blur_ig.GetMask(
  test_image_single[0], feed_dict = {neuron_selector: 1})

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(blur_ig_mask_grayscale, title='Blur Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))


plt.show()
