## DeepBleed

ICH volumetric estimation is a task routinely performed in clinical practice. This is the first publicly available deep neural network model to perform the task of ICH segmentation and volume estimation. The model was originally developed in testing the hypothesis that an ICH segmentation deep neural network could be trained in an earlier clinical trial phase (MISTIE Phase II) and validated by a later phase (MISTIE Phase III), with results on par or better than customized architectures and models trained on significantly larger, curated single center datasets. The implication being that such a model could be used to derive metrics for a multicenter clinical trial. 

We provide the 3D model from our paper "High-Throughput 3D Segmentation of Intracerebral Hemorrhage: Development and Validation within a Clinical Trial Series". The model version 1.0 will perform binary segmentation of ICH and will include areas of IVH if present. The original model expects non-contrast CT with prior preprocessing described in our paper, including a validated brain extraction method and spatial normalization via registration to a 1.5mm x 1.5mm x 1.5mm template with a shape of (128, 128, 128). Support for deployment on a local machine or on the cloud via docker image is provided. 

Please read the software license, this is not intended for any clinical or commercial use. 


To install move to the directory of DeepBleed that was just cloned and type:
```
$ python setup.py install
```
If it happens to be missing some dependencies listed above, you may install them with pip: <br/>
```
$ pip install tensorflow-gpu==2.0.0
$ pip install nibabel
$ ...
```
Alternatively, you can run the program in docker. You can directly pull the docker image from dockerhub and avoid installing dependencies. (For tutorials on docker, see [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

```
$ docker pull msharrock/neuroimage:tf-2.0
$ docker run -it --rm msharrock/neuroimage:tf-2.0
```
The underlying deep neural network architecture is based on the VNET by Milletari et al. at https://github.com/faustomilletari/VNet



## Authors

* **Matthew Sharrock** - *Study Design, Neural Network Dev/Training*
* **John Muschelli** - *Study Design, Preprocessing, Statistical Validation*

