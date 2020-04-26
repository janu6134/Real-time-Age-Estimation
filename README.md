The weight files can be downloaded from https://drive.google.com/open?id=1CdmPGVtWD1H0HzHLxBIImHXbzv-uAqLx
Put the three weight files individually in the project folder. Do not organize them in a single folder, or else path issues will come up.

To execute the demo, run :
python demo.py --model_name modelname --weight_file weightfile.hdf5 


In this command, 'modelname' refers to either AlexNet, ResNet50 or WideResNet. 'weightfile.hdf5' refers to the corresponding weight file downloaded from the google drive link. 
For alexnet, it is alexnet.hdf5; for resnet50 it is resnet50.hdf5 and for wideresnet it is wideresnet.hdf5
