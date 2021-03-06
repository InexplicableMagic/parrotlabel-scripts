#!/usr/bin/env python

#####################################################################################################################################
#
# Convert ParrotLabel native JSON image annontation format to a TensorFlow binary tfrecords format training examples file.
# Also produces a TensorFlow label map (pbtxt) file
# A validation examples file can also optionally be produced
#
# Intended as an input to the TensorFlow Object Detection API
#
# For use with ParrotLabel available from: https://github.com/InexplicableMagic/parrotlabel
#
# Usage Example: 
#
# - Split a ParrotLabel JSON annotations file into 80% training and 20% validation (randomising the order)
# - Writes the training image annontations to a TensorFlow TFRecord training file at "training_file.records"
# - Writes the validation image annontations to a TensorFlow TFRecord training file at "validation.records"
# - The original images are located in the "images_dir" directory
#
# ./parrotlabel_to_tfrecords.py \
#			--tfrecords training_file.records \
#			--val_tfrecords validation.records \
#			--validation_percentage 20 \
#			--labelmap labelmap.pbtxt \
#			--images images_dir/ \
#			input_parrot_annotations.json
#
# The validation output is optional. Omitting the "val_tfrecords" option will writes all of the available annotations to the training
# examples files.
#
# For use with JPEG and PNG format images only
#
#####################################################################################################################################

import argparse
import tensorflow as tf
import json
import os
import io
import sys
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import math
import random

max_index=0
labelIndex = dict()
images_seen = 0

def getIndexofLabel(label):
	global max_index
	if label in labelIndex:
		return labelIndex[ label ]
	else:
		max_index+=1
		labelIndex[ label ] = max_index
		return labelIndex[ label ]

def generateLabelMap(fname):
	msg = StringIntLabelMap()
	labels_sorted = sorted(labelIndex.items(), key=lambda item: item[1])
	for label in labels_sorted:
		msg.item.append(StringIntLabelMapItem(id=label[1], name=label[0]))
	text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
	with open(fname, 'w') as f:
	        f.write(text)		
	

def resolve_images_dir( json, json_dir ):
	if "config"  in json:
		if "image_dir_base_path" in json['config']:
			d = json['config']['image_dir_base_path']
			if os.path.isdir( d ):
				return d
			joined_dir = os.path.join( json_dir, d )
			if os.path.isdir(joined_dir  ):
				return joined_dir
			
	return json_dir

def generate_label_list(label_list, json, images_dir_base):
	global images_seen
	labels = json['labels'];
	for label in labels:
		if "image_path" in label:
			full_path = os.path.join( images_dir_base, label['image_path'] )
			images_seen+=1
			if os.path.isfile( full_path ):
				if "box_list" in label and len(label['box_list']) > 0:
					label['full_path'] = full_path
					label_list.append(label)
				else:
					print("INFO: Skipping image as it has no annotations:"+full_path)
			else:
				print("WARN: Skipping missing image at path:"+full_path)
	return label_list
	

def write_tf_records(label_list, tfwriter):
	examples_written = 0
	for label in label_list:
		with tf.io.gfile.GFile(label['full_path'], 'rb') as image_fp:
			encoded_image_bytes = image_fp.read()
			encoded_image_io = io.BytesIO(encoded_image_bytes)
			image = Image.open(encoded_image_io)
			(image_width, image_height) = image.size
		
			filename =  label['image_path'].encode('utf8')
			image_format = b'jpg'
			if label['image_path'].lower().endswith(".png"):
				image_format = b'png'

			xmins = []
			xmaxs = []
			ymins = []
			ymaxs = []
			classes_text = []
			classes = []
		
		
			for box in label['box_list']:
				xmin = math.floor( image_width * box['leftPct'] )+1
				boxWidth = round( image_width * box['widthPct'] )-1
				ymin = math.floor( image_height * box['topPct'] )+1
				boxHeight = round( image_height * box['heightPct'] )-1
				if(boxWidth < 0):
					boxWidth = 0;
				if(boxHeight < 0):
					boxHeight = 0;
				xmax = xmin+boxWidth;
				ymax = ymin+boxHeight;
				xmins.append(float(xmin) / image_width)
				xmaxs.append(float(xmax) / image_width)
				ymins.append(float(ymin) / image_height)
				ymaxs.append(float(ymax) / image_height)
				classes_text.append(box['label'].encode('utf8'))
				classes.append(getIndexofLabel( box['label'] ))
			
				binary_record = tf.train.Example(features=tf.train.Features(feature={
					'image/height': dataset_util.int64_feature(image_height),
					'image/width': dataset_util.int64_feature(image_width),
					'image/filename': dataset_util.bytes_feature(filename),
					'image/source_id': dataset_util.bytes_feature(filename),
					'image/encoded': dataset_util.bytes_feature(encoded_image_bytes),
					'image/format': dataset_util.bytes_feature(image_format),
					'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
					'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
					'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
					'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
					'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
					'image/object/class/label': dataset_util.int64_list_feature(classes),
				}))
				tfwriter.write(binary_record.SerializeToString())
				examples_written+=1
				
	return examples_written
	
	

	
def load_parrotlabel_json(json_fpath):
	if os.access(json_fpath, os.R_OK):
		with open( json_fpath ) as json_fp:
			data = []
			try:
				data = json.load(json_fp)
			except Exception as e:
				sys.exit("Failed to parse file \""+json_fpath+"\" as valid JSON: "+str(e))
			
			if data:
				if "file_format" in data and "magic" in data['file_format']:
					if data['file_format']['magic'] == "43a04f6d-f95b-41da-8b92-f4c9f859d3fb":
						if "labels" in data:
							return data
						else:
							sys.exit("File \""+json_fpath+"\" is not valid. Labels section missing.")						
					else:
						sys.exit("File \""+json_fpath+"\" is not a valid ParrotLabel file. Magic key has incorrect value.")	
				else:
					sys.exit("File \""+json_fpath+"\" is not a valid ParrotLabel file. No magic key.")
			
			
		
	else:
		sys.exit("Cannot read file:"+json_fpath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Convert parrotlabel JSON to TensorFlow tfrecords")
	parser.add_argument("--images", type=str, help="Directory where images are stored")
	parser.add_argument("--labelmap", type=str, help="Set the name of the label map (.pbtxt) file to generate", required=True)
	parser.add_argument("--tfrecords", type=str, help="Set output tfrecords filename", required=True)
	parser.add_argument("--val_tfrecords", type=str, help="Filename in which to write a validation set", required=False)
	parser.add_argument("--validation_percentage", type=int, help="Split out this percentage of records for validation", required=False)
	parser.add_argument('json_paths', nargs='*')
	args = parser.parse_args()
	
	if(len(args.json_paths) < 1):
		sys.exit("At least one parrotlabel JSON file must be supplied");
	
	val_pct = 20
	if args.validation_percentage is not None:
		if( args.validation_percentage < 1 or args.validation_percentage > 99 ):
			sys.exit("Validation_percentage must be between 1 and 99")
		else:
			val_pct = args.validation_percentage
	
	images_dir_base = "./"
	if args.images is not None:
		images_dir_base = args.images
		if not os.path.isdir(images_dir_base):
			sys.exit("Images directory does not exist:"+images_dir_base);

	label_list = []
	for json_path in args.json_paths:
		json_data = load_parrotlabel_json( json_path )
		if args.images is None:
			images_dir_base = resolve_images_dir( json_data, os.path.dirname(os.path.abspath(json_path)) )
		generate_label_list( label_list, json_data, images_dir_base )
	
	val_examples_written = 0
	training_examples_written = 0
	if( len(label_list) > 0 ):	
		training_label_set = label_list
		validation_label_set = []
		if( args.val_tfrecords is not None ):
			val_size = int(len( label_list )*(val_pct/100))
			random.shuffle( label_list )
			validation_label_set = label_list[0:val_size]
			training_label_set = label_list[val_size:]
			tfwriter = tf.io.TFRecordWriter(args.val_tfrecords)
			val_examples_written = write_tf_records( validation_label_set, tfwriter )
			tfwriter.close()	

		tfwriter = tf.io.TFRecordWriter(args.tfrecords)
		training_examples_written = write_tf_records( training_label_set, tfwriter )
		tfwriter.close()
	
	generateLabelMap( args.labelmap )
	print(	"Images Seen:"+str(images_seen)+
		"\nTraining Images Written:"+str(len(training_label_set))+
		"\nTraining Boxes Written:"+str(training_examples_written)+
		"\nValidation Images Written:"+str(len(validation_label_set))+
		"\nValidation Boxes Written:"+str(val_examples_written)+
		"\nUnique Label Categories:"+str( len(labelIndex) ), file=sys.stderr)
	
	

