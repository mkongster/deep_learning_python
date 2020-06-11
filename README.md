# deep_learning_python
deep learning using python with keras/tensorflow

Prerequsites (UNIX, Python3):

    Install BLAS Library (Open BLAS)
        sudo apt-get install build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev

    Install Numpy Scipy, Matplotlib  (Python3)
        sudo apt-get install python3-numpy python3-scipy python3-matplotlib python3-yaml

    Install HDF5
        sudo apt-get install libhdf5-serial-dev python3-h5py

    Install Graphviz and pydot-ng
        sudo apt-get install graphviz 
        sudo pip3 install pydot-ng
    
    Install additional packages used in Deep Learning with Python book
        sudo apt-get install python3-opencv 

    Setting up GPU support (optional), go to nvidia for the latest CUDA, cnDNN libraries
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 
        sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb 
        sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb
        sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
        sudo apt-get -y install cuda 

        sudo dpkg -i libcudnn8*.deb

    Install TensorFlow (with or without gpu support)
        sudo pip3 install tensorflow-gpu
        sudo pip3 install tensorflow

    Install Keras
        sudo pip3 install keras