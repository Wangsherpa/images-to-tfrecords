import os
import tensorflow as tf
from tensorflow.train import BytesList, Int64List
from tensorflow.train import Example, Features, Feature
from tqdm import tqdm

from contextlib import ExitStack

def create_example(image, label, class_map):
    # print("Label", label.numpy())
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
    image = tf.io.read_file(imgPath)
    image = tf.image.decode_png(image)

    # grab the label
    label = tf.strings.split(imgPath, os.path.sep)[-2]
    return (image, label)

