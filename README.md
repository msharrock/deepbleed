## DeepBleed

ICH volumetric estimation is a task routinely performed in clinical research. This is the first publicly available deep neural network model to perform the task of ICH segmentation and volume estimation. The model expects non-contrast CT in NIfTI format and automatically performs preprocessing including a validated CT brain extraction method and spatial normalization via registration to a 1.5mm x 1.5mm x 1.5mm CT template with a shape of (128, 128, 128). 
 <br/>


Current dependencies are listed below. 

Software | Version | URL
------------ | ------------- | -------------
Tensorflow | gpu-2.0.0-rc0 | https://www.tensorflow.org
ANTsPy | 0.1.4 | https://github.com/ANTsX/ANTsPy
FSLPy | 2.7.0 | https://git.fmrib.ox.ac.uk/fsl/fslpy
FSL | 6.0.2 | https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

<br/>

To install on a local debian linux machine, cd into the deepbleed directory and run the setup script:<br/>
```
$ python3 setup.py 
```
<br/>
To run an ICH prediction set the path to directories for inputs, outputs and model weights:<br/>
```
$ python3 predict.py --indir /path/to/inputs/ --outdir /path/to/outputs/ --weights /path/to/weights
```
You may optionally specify the number of GPUs and CPUs to use with --gpus and --cpus.
<br/>

Alternatively, you can pull a pre-built docker image with the dependencies installed:<br/>
```
$ docker pull msharrock/deepbleed
```
<br/>
To run a prediction, start the docker image and initiatize with access to the data path volume:<br/>
```
$ docker run -it msharrock/deepbleed bash -v /path/to/data
```
<br/> 
Then run predictions as previously noted, the working directory on launch is set to this repository
