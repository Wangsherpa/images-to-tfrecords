# images-to-tfrecords v1.0
Converting Images to TFRecords

# How to use?
```
git clone https://github.com/Wangsherpa/images-to-tfrecords.git

python create_tfrecs.py --dataset data/train --savepath data/TFRecords --prefix train --n_shards 10
```

This will create 10 TFRecord files with names:
- train-1.tfrecord	
- train-2.tfrecord	
- train-3.tfrecord	
- train-4.tfrecord	
- train-5.tfrecord	
- train-6.tfrecord	
- train-7.tfrecord	
- train-8.tfrecord	
- train-9.tfrecord	
- train-10.tfrecord

# Directory Structure:
<img src="/images/directory_screenshot.png" height=400>

**Command line arguments (with respect to above directory tree)**:
- **--dataset**: data/train or data/test
- **--savepath**: can be a new directory to save TF Record files
- **--prefix**(optional): train (default="file")
- **--n_shards**: 
  - auto: number of shards required will be computed automatically
  - integer: number of shards to be created
