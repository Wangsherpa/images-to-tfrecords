# images-to-tfrecords
Converting Images to TFRecords

# How to use?
```
git clone https://github.com/Wangsherpa/images-to-tfrecords.git

python create_tfrecs.py --dataset data/train --savepath data/TFRecords/train --n_files 10
```

This will create 10 TFRecord files with names:
- train.tfrecord-00001-of-00020	
- train.tfrecord-00002-of-00020	
- train.tfrecord-00003-of-00020	
- train.tfrecord-00004-of-00020	
- train.tfrecord-00005-of-00020	
- train.tfrecord-00006-of-00020	
- train.tfrecord-00007-of-00020	
- train.tfrecord-00008-of-00020	
- train.tfrecord-00009-of-00020	
- train.tfrecord-00010-of-00020

# Directory Structure:
<img src="/images/directory_screenshot.png" height=400>
