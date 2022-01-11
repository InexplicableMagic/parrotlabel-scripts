# ParrotLabel Scripts

Utility scripts for use with the [ParrotLabel](https://github.com/InexplicableMagic/parrotlabel) graphical image annotation tool. For creating training data for object detection and other machine learning applications.

## parrotlabel_to_tfrecords

A script to convert the native ParrotLabel JSON format into the TensorFlow [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. Also produces an associated Label Map (pbtxt) file. These files can be used as an input to the TensorFlow Object Detection API for training.

Usage Example:

* Split a ParrotLabel JSON annotations file into 80% training and 20% validation (randomising the order)
* Writes the training image annontations to a TensorFlow TFRecord training file at "training_file.records"
* Writes the validation image annontations to a TensorFlow TFRecord training file at "validation.records"
* Writes a TensorFlow Label Map file to "labelmap.pbtxt"

```
 ./parrotlabel_to_tfrecords.py \
			--tfrecords training_file.records \
			--val_tfrecords validation.records \
			--validation_percentage 20 \
			--labelmap labelmap.pbtxt \
			--images images_dir/ \
			input_parrot_annotations.json
```

Options:

* --tfrecords - The file name to output the TensorFlow Records training file to
* --val_tfrecords - Optionally a file to write out a proportion of the annotations as validation examples (also in TFRecords format)
* --validation_percentage - What percentage of the annotations to use as validation example (default is 20%)
* --labelmap - The file name at which to output the associated TensorFlow label map file (pbtxt)
* --images - The path to the directory where the images are stored (use only JPEG or PNG images)

Followed by one or more ParrotLabel JSON format annotation files as arguments.

## example_annotations

An example ParrotLabel JSON annotations file for testing scripts. Includes several annotated images of parrots.
