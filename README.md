# p23-disentangled-medical-images
Official repository of the paper "Anonymizing medical case-based explanations through disentanglement"

## Requirements

* tensorflow (version: 1.14.0 or above)
* scikit-learn (version: 1.2.1)
* h5py (version: 3.1.0)

## Data

The data that is provided to the models is organized in a ```.hdf5``` file with the following folders:

**id** - identity annotations, where each image is assigned a numerical value corresponding to the ID of the patient \
**dis** - medical annotations, where each image is assigned a value in [0, number of different commorbities[ \
**set** - 0 if the image belongs to the training set, 1 for the test set or and 2 for the validation set \
**images** - array of images


## Models

### Disentanglement network with multi-class identity recognition: ```siamese``` folder
  * Training and testing of disentanglement network available in ```siamese/disentanglement_network.py```
  * Training and testing of anonymization network available in ```siamese/anonymization_network.py```
  * At the end of each epoch, an image that shows the current results of the respective model on a test case is saved on folder ```images```
  * Model weights are saved on folder ```models``` 


### Disentanglement network with siamese identity recognition: ```multiclass``` folder
  * Training and testing of disentanglement network available in ```multiclass/disentanglement_network.py```
  * Training and testing of anonymization network available in ```multiclass/anonymization_network.py```
  * At the end of each epoch, an image that shows the current results of the respective model on a test case is saved on folder ```images```
  * Model weights are saved on folder ```models``` 
