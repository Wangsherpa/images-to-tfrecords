import os
import tensorflow as tf
from tensorflow.train import BytesList, Int64List
from tensorflow.train import Example, Features, Feature
from tqdm import tqdm
import cv2

from contextlib import ExitStack

def create_example(image, label, class_map):
    """
    Function to serialize each record.

    Args:
        image: image to serialize
        label: image's corresponding label
        class_map (dict): class IDs
    
    Returns:
        dict: Features

    """
    image = tf.io.serialize_tensor(image)
    return Example(
        features=Features(
            feature={
                'image': Feature(bytes_list=BytesList(value=[image.numpy()])),
                'label': Feature(int64_list=Int64List(value=[
                    class_map[tf.compat.as_str_any(label.numpy())]
                ])),
            }
        )
    )

def write_tfrecords(name, dataset, n_shards=10, class_map=None):
    """
    This function saves a given dataset to a set of TFRecord files.
    
    Args:
            name (string) path to save target files along with initial name
            dataset (tf.data) tensorflow dataset
            n_shards (int) total number of tfrecord files to create
            class_map (dict) contains class IDs
    
    Returns:
            list: list of new tfrecord file paths
    """
    paths = ["{}.tfrecord-{:05d}-of-{:05d}".format(name, index+1,
             n_shards) for index in range(n_shards)]
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                   for path in paths]
        # print("[INFO] creating tfrecord files...")
        for index, (image, label) in tqdm(dataset.enumerate()):
            shard = index % n_shards
            example = create_example(image, label, class_map=class_map)
            writers[shard].write(example.SerializeToString())
        return paths

@tf.autograph.experimental.do_not_convert
def load_images(imgPath):
    """
    Function to load image and extract label.

    Args:
        imgPath: path of a image
    
    Returns:
        tuple: (image, label)
    """
    image = tf.io.read_file(imgPath)
    image = tf.image.decode_png(image)

    # grab the label
    label = tf.strings.split(imgPath, os.path.sep)[-2]
    return (image, label)

# compute number of shards based on total image size
def compute_nshards(image_paths):
    print("[INFO] Computing number of shards required.")
    total_image_size = 0
    for path in image_paths:
        image = cv2.imread(path)
        total_image_size += image.nbytes / (1024 * 1024.0)
    # calculate no of shards required
    # divide by 150MB (resulting file will be between 150-200 Mega Bytes)
    n_shards = total_image_size // 150
    return n_shards