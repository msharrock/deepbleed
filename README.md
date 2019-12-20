## DeepBleed

ICH volumetric estimation is a task routinely performed in clinical practice. This is the first publicly available deep neural network model to perform the task of ICH segmentation and volume estimation. The model was originally developed to test the hypothesis that an ICH segmentation deep neural network could be trained in an earlier clinical trial phase (MISTIE Phase II) and validated by a later phase (MISTIE Phase III).

We provide the 3D model from our paper "3D Deep Neural Network Segmentation of Intracerebral Hemorrhage: Development and Validation within a Clinical Trial Series". This model will perform binary segmentation of ICH and will include areas of IVH if present. The original model expects non-contrast CT in NIfTI format with prior preprocessing described in our paper, including a validated brain extraction method and spatial normalization via registration to a 1.5mm x 1.5mm x 1.5mm template with a shape of (128, 128, 128) which are all provided.  


If it happens to be missing some dependencies listed above, you may install them with pip: <br/>
```
$ pip install tensorflow-gpu==2.0.0
$ pip install fslpy
$ pip install antspy
$ pip install nibabel
$ ...
```
The underlying deep neural network architecture is based on the VNet by Milletari et al. at https://github.com/faustomilletari/VNet



## Authors

* **Matthew Sharrock** - *Study Design, Neural Network Dev/Training, Preprocessing(Python)*
* **John Muschelli** - *Study Design, Preprocessing, Statistical Validation*

