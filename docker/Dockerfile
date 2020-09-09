ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

# Choose the Python version
ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# Choose the CUDA and TF Versions
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=2.1.0

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas10=10.2.1.243-1 \ 
        libcublas-dev=10.2.1.243-1 \
        cuda-nvrtc-${CUDA/./-} \
        cuda-nvrtc-dev-${CUDA/./-} \
        cuda-cudart-dev-${CUDA/./-} \
        cuda-cufft-dev-${CUDA/./-} \
        cuda-curand-dev-${CUDA/./-} \
        cuda-cusolver-dev-${CUDA/./-} \
        cuda-cusparse-dev-${CUDA/./-} \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libcudnn7-dev=${CUDNN}+cuda${CUDA} \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-${CUDA}/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/${LIB_DIR_PREFIX}-linux-gnu/libcudnn_static_v7.a

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-dev=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin-dev=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# Configure the build for our CUDA configuration.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/include/x64_64-linux-gnu:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_VERSION=${CUDA}
ENV TF_CUDNN_VERSION=${CUDNN_MAJOR_VERSION}
# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    #python3.7 \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

#python binary if needed
RUN ln -s $(which python3) /usr/local/bin/python

# install tensorflow
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

#install utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    virtualenv \
    swig 

#install useful packages
RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    enum34 \
    #imaging packages
    nibabel \
    six \
    scikit-image \
    webcolors \
    plotly \
    webcolors \
    fslpy

RUN ${PIP} install antspyx
#https://github.com/ANTsX/ANTsPy/releases/download/v0.2.0/antspyx-0.2.0-cp37-cp37m-linux_x86_64.whl


#Install FSL base, but not unneeded extras
ENV INSTALL_FOLDER=/usr/local/

RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.2-centos7_64.tar.gz | tar xz -C ${INSTALL_FOLDER} \
    --exclude='fsl/doc' \
    --exclude='fsl/data/first' \    
    --exclude='fsl/data/atlases' \
    --exclude='fsl/data/possum' \    
    --exclude='fsl/src' \    
    --exclude='fsl/extras/src' \    
    --exclude='fsl/bin/fslview*' \
    --exclude='fsl/bin/FSLeyes' \
    --exclude='fsl/bin/*_gpu*' \
    --exclude='fsl/bin/*_cuda*'

# Configure environment
ENV FSLDIR=${INSTALL_FOLDER}/fsl/ \
    FSLOUTPUTTYPE=NIFTI_GZ

# Below needs a new line
ENV PATH=${FSLDIR}/bin:$PATH \
    LD_LIBRARY_PATH=${FSLDIR}:${LD_LIBRARY_PATH}

RUN mkdir /.local && chmod a+rwx /.local
RUN git clone https://github.com/msharrock/deepbleed && chmod a+rw ./deepbleed

WORKDIR /deepbleed

# download the weights
RUN wget -O weights.zip https://www.dropbox.com/s/v2ptd9mfpo13gcb/mistie_2-20200122T175000Z-001.zip?dl=1  && unzip -j weights.zip 
RUN for i in _data-00001-of-00002 _data-00000-of-00002 _index; do out=`echo ${i} | sed "s/_/weights./"`; mv ${i} ${out}; done

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc"]
