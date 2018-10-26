from lime import lime_image
import time
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sys
import numpy as np
from skimage.segmentation import mark_boundaries
import re
import os

image_path = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

def predict_fn(images):
    all_predictions = []

    # Unpersists graph from file
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for image_arr in images:
            cv2.imwrite("tmp.jpeg", image_arr)
            image_data = tf.gfile.FastGFile('tmp.jpeg', 'rb').read()

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            all_predictions.append(predictions[0, :])
    return np.array(all_predictions)


if __name__ == "__main__":
    # Reading the image and changing the colours to the right format (RGB BGR etc.)
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    explainer = lime_image.LimeImageExplainer(feature_selection="none")

    tmp = time.time()
    print("Starting Flips")
    # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
    #### IMPORTANT THIS IS WHERE IMAGE IS BASED ON; set samples == batch size for speed and more samples in general means a better explanation
    explanation = explainer.explain_instance(im, predict_fn, hide_color=0, num_samples=1000, batch_size=1000)
    print("Time needed: ", time.time() - tmp)

    # What do we predict:
    model_prediction = predict_fn([im])
    # Which index has highest score
    max_index = model_prediction.argmax()

    # Reading labels
    f = open(label_path , "r")
    text = f.read().splitlines()
    f.close()

    # Creating a lime image based on all these predictions (slight pixel flips and predictive outcomes)
    temp, mask = explanation.get_image_and_mask(max_index, positive_only=False, num_features=10, hide_rest=False, min_weight=0)

    plt.imshow(mark_boundaries(temp, mask))

    # Name of the file + what we predict and the confidence we have in this
    plt.title('{}\npred: {}\nconf: {:.2f}'.format(re.split('[./]', image_path)[1],text[max_index], model_prediction[0][max_index]))
    plt.axis('off')

    plt.savefig('{}_LIME_{}.png'.format(re.split('[./]', image_path)[1],text[max_index]))
    plt.close()
    os.remove('tmp.jpeg')