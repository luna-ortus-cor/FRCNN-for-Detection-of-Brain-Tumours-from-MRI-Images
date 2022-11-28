This is an implementation of various faster region-based convolutional neural networks (henceforth FR-CNN) to detect presence of tumours from brain MRI scans.

This implementation was done using tensorflow-1.15.2-gpu on Google Colaboratory.

Some things to note:
a. this will not work with tensorflow-2.0 or higher.
b. if not running on Google Colaboratory, %%bash cell magics may not work, an alternative is required. 
c. it is recommended to have at least 25GB of available disk space per model.
d. (optional) you may wish to have another Google Colaboratory .ipynb open to run tensorboard on the model directory for better visualisation

In this folder you will find several subfolders.

1. model_data contains 2 subfolders, stage 1 and stage 2. the data in stage 1 is not in a usable file structure, user will have to reprocess data or manually format file structure
1a. stage 2 contains 2 subfolders, all_img_final and xmls, and 4 files, csv_final.csv, test_keys.txt, train_keys.txt, val_keys.txt
1b. all_img_final contains 16508 jpg images, each representing one slice of brain MRI scan
1c. xmls contains 3 folders (testxml, trainxml, valxml) and 6 files (testxml.csv, testxml.record, trainxml.csv, trainxml.record, valxml.csv, valxml.record)
1ci. testxml contains 2004 xml files, each corresponding to 1 image in the test set
1cii. trainxml contains 12500 xml files, each corresponding to 1 image in the train set
1ciii. valxml contains 2004 xml files, each corresponding to 1 image in the val set
1civ. each of test, train, val have a corresponding csv file representing the list of xmls in the set, and a record file representing the tfrecord for the set
1d. csv_final.csv is a compilation of all the csvs from the individual datasets we used
1e. the 3 .txt files are provided as an example of the train-val-test split we used. each .txt file contains the image file names in that set

2. 6 folders of the format model_frcnn_[name]_full. These are the models from stage 1 of training. 
2a. each folder contains 1 model, including the initial weights, training weights, frozen graph of each checkpoint, and evaluation
2bi. the models used are 
2bii. inception_resnet_v2_2, the inception resnet v2, initial learning rate 0.0001
2biii. inception_resnet_v2, the inception resnet v2, initial learning rate 0.001
2biv. inception_v2, the inception v2
2bv. resnet50rpn2, the resnet50, rpn at block 2
2bvi. resnet50rpn3, the resnet50, rpn at block 3
2bvii. resnet101rpn3, the resnet101, rpn at block 3
2c. The total file size of these 6 models is 81GB

3. 3 folders of the format model_frcnn_[name]_final_full. These are the models from stage 2 of training.
3a. each folder contains 1 model, including the initial weights, training weights, frozen graph of each checkpoint, and evaluation
3bi. the models used are 
3bii. inception_resnet_v2_2, the inception resnet v2, trained starting weights, initial learning rate 0.0001
3biii. inception_resnet_v2, the inception resnet v2, trained starting weights, initial learning rate 0.001
3biv. inception_resnet_v2_uninitialized, the inception resnet v2, untrained starting weights, initial learning rate 0.001
3c. the total file size of these 3 models is 28GB

4. model_results 
4a. This directory contains all the results of our model
4b. Each model has its own csv which contains mAP, mAR, F1 score at every checkpoint
4c. result.docx stores all results in word form, the 6 models from stage 1 training have the best checkpoint highlighted
4d. results.xlsx stores the best results for all models in excel form
4e. sample_detections directory stores example detections made by our best model from stage 2 (10 of them, in .png format)

5. pre-post-processing, containing the source code in .ipynb files
5ai. in chronological order to run, 
5aii. brats_hgg_file_processing.ipynb
5aiii. brats_lgg_file_processing.ipynb
5aiv. decathlon_file_processing.ipynb
5av. images_processing.ipynb
5avi. tcga_file_processing.ipynb
5avii. xml labels.ipynb
5aviii. frcnn.ipynb
5aix. frozen_graph.ipynb
5ax. actual_inference.ipynb

In order to use:
Choose the method which best suits your needs, from 1A, 1B, 1C, 1D

1A. Running from scratch and with external dataset
1Ai. obtain the dataset you wish to use
1Aii. process the data, obtaining 
1Aiia. the image (in jpg format) and its file name/path
1Aiib. x and y coordinates of the top left corner and bottom right corner of the bounding box
1Aiic. width and height of the image
1Aiid. class label of the image
1Aiii. depending on the format of the data, it may or may not be possible to use the data processing .ipynb we include here
1Aiv. write the above data to xml in pascal-voc format, and create csv for each of train test val set
1Av. convert the xml to tfrecord

1B. Running from scratch and with the pre-included dataset
1Bi. download the pre-included dataset

1C. Using a pre-trained model with external dataset
1Ci. obtain the dataset you wish to use
1Cii. process the data, obtaining 
1Ciia. the image (in jpg format) and its file name/path
1Ciib. x and y coordinates of the top left corner and bottom right corner of the bounding box
1Ciic. width and height of the image
1Ciid. class label of the image
1Ciii. depending on the format of the data, it may or may not be possible to use the data processing .ipynb we include here
1Civ. write the above data to xml in pascal-voc format, and create csv for each of train test val set
1Cv. convert the xml to tfrecord
1Cvi. download one of the pre-trained models. for best results we recommend using the inception_resnet_v2 model.

1D. Using a pre-trained model with the pre-included dataset
1Di. download one of the pre-trained models. for best results we recommend using the inception_resnet_v2 model.
1Dii. download the pre-included dataset

To train, run frcnn.ipynb. Ensure all paths point to correct directories and files
After training, run frozen_graph.ipynb on each checkpoint to export the model as a frozen_graph.pb
After exporting the model, run actual_inference.ipynb on each exported model

The file structure should resemble the following
root
|-models
|-images
|-model_dir
    |-training
    |-frozen
    |-saved_model
    |-eval
|-xmls
    |-trainxml
    |-testxml
    |-valxml
    |-trainxml.csv
    |-trainxml.record
    |-testxml.csv
    |-testxml.record
    |-valxml.csv
    |-valxml.record
|-frcnn.ipynb
|-frozen_graph.ipynb
|-actual_inference.ipynb
|-any other files necessary

:D