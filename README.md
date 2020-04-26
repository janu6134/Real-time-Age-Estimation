The file 'model.py' contains the code for AlexNet and ResNet50. The file 'wide_resnet.py' contains code for WideResNet architecture.
Before training the models, we need to prepare the dataset. In this project, the UTKFace dataset and APPA-REAL dataset are used together.
You can download the UTKFace dataset using this link https://susanqq.github.io/UTKFace/. Make sure to download the folder 'In-the-wild Faces'. Extract the tar.gz files 'part1', 'part2' and 'part3', and pre-process them by running the following command:
python utkdataset.py --input_dir inputdir --output_dir outputdir

In the above command, 'inputdir' refers to the directory where your extracted folders are present. 'outputdir' refers to the destination folder where your utkfaces will be placed.
After preparing the utkface dataset, training can be started. The script 'dataloader.py' contains code to process images from both appa-real and utkface for training and validation, and store them in a single list for creating the dataloaders in train.py.
For training, run the following command :
python train.py --model_name modelname --n_epoch nepoch --appareal_data path_to_appareal --utk_data path_to_utk

In the above command, 'modelname' refers to either AlexNet, ResNet50 or WideResNet; 'nepoch' refers to the number of epochs you want to train for; 'path_to_appareal' and 'path_to_utk' refer to the path of the directories where the datasets are present.


The weight files can be downloaded from https://drive.google.com/open?id=1CdmPGVtWD1H0HzHLxBIImHXbzv-uAqLx
Put the three weight files individually in the project folder. Do not organize them in a single folder, or else path issues will come up.

To execute the demo, run :
python demo.py --model_name modelname --weight_file weightfile.hdf5 


In this command, 'modelname' refers to either AlexNet, ResNet50 or WideResNet. 'weightfile.hdf5' refers to the corresponding weight file downloaded from the google drive link. 
For alexnet, it is alexnet.hdf5; for resnet50 it is resnet50.hdf5 and for wideresnet it is wideresnet.hdf5
