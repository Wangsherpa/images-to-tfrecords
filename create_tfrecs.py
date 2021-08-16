import os
import argparse
import tensorflow as tf
from imutils import paths
from tfrecord_utils import write_tfrecords
from tfrecord_utils import load_images, compute_nshards


# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset")
ap.add_argument("-p", "--prefix", default="file",
                help="path to dataset") 
ap.add_argument("-s", "--savepath", required=True,
                help="path to save tfrecord files")
ap.add_argument("-n", "--n_shards", default=10,
                help="total number of TFRecord files to generate")
args = vars(ap.parse_args())

# grab the list of images in our dataset directory
image_paths = list(paths.list_images(args['dataset']))
dir_items = sorted(os.listdir(args['dataset']))
class_names = [path for path in dir_items if not path.startswith(".")]
class_map = {class_: id_ for id_, class_ in enumerate(class_names)}

# create tensorflow dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_images, num_parallel_calls=AUTOTUNE)

# write to tfrecord files
if args['n_shards'] == 'auto':
    n_shards = int(compute_nshards(image_paths))
    print("[INFO] {} shards will be created.".format(n_shards))
else:
    n_shards = int(args['n_shards'])
new_paths = write_tfrecords(args["savepath"], dataset, prefix=args['prefix'],
                            n_shards=n_shards, class_map=class_map)
print("[INFO] Class IDs: {}".format(class_map))