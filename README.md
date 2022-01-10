# ParrotLabel Scripts

Utility scripts for use with the [ParrotLabel](https://github.com/InexplicableMagic/parrotlabel) graphical image annotation tool. For creating training data for object detection and other machine learning applications.

## parrotlabel_to_tfrecords

A script to convert the native ParrotLabel JSON format into the TensorFlow [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format and also produces an associated Label Map (pbtxt) file. These files can be used as an input to the TensorFlow Object Detection API for training.

Usage Example:

```
./parrotlabel_to_tfrecords.py --tfrecords output_tf_file.records --labelmap output_labelmap.pbtxt --images images_dir/ input_parrot_annotations.json
```

Options:

* --tfrecords - The file name to output the TensorFlow Records file to
* --labelmap - The file name at which to output the associated TensorFlow label map file (pbtxt)
* --images - The path to the directory where the images are stored (only JPEG or PNG images)

Followed by one or more ParrotLabel JSON format annotation files as arguments.

## example_annotations

A example ParrotLabel JSON annotations file for testing scripts. Includes several annotated images of parrots.
