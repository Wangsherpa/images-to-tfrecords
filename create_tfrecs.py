import os
import argparse
import tensorflow as tf
from imutils import paths
from tfrecord_utils import create_example, write_tfrecords
from tfrecord_utils import load_images

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset")
ap.add_argument("-p", "--savepath", required=True,
                help="path to save tfrecord files")
args = vars(ap.parse_args())

# grab the list of images in our dataset directory
image_paths = list(paths.list_images(args['dataset']))
class_names = sorted(os.listdir(args['dataset']))
class_map = {class_: id_ for id_, class_ in enumerate(class_names)}

# create tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_images)

new_paths = write_tfrecords(args["savepath"], dataset, n_shards=10, class_map=class_map)
print("[INFO] Class IDs: {}".format(class_map))

    
