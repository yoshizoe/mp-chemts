# Large Scale Distributed Parallel MCTS algorithm
This project developed highly scalable distributed parallel MCTS algorithms and their applications for molecular design probelm. 
Check the paper for details: [Practical Large-Scale Distributed Parallel Monte-Carlo Tree Search Applied to Molecular Design.](https://arxiv.org/abs/2006.10504)
# Requirements
The code was tested on Linux and MacOS, we recommend using anaconda to install the following softwares.
1. [Python](https://www.anaconda.com/products/individual)(version 3.7.4)
2. [MPI](https://anaconda.org/conda-forge/openmpi)
3. [mpi4py](https://anaconda.org/anaconda/mpi4py)(version 3.0.3)
4. [RDkit](https://anaconda.org/rdkit/rdkit)
5. [Keras](https://keras.io/about/)(version 2.0.5)
6. [Tensorflow](https://www.tensorflow.org/install/pip)(verison 1.15.2)
7. [Nltk](https://anaconda.org/anaconda/nltk)
8. [Networkx](https://anaconda.org/anaconda/networkx)
9. [Gaussian 16](http://gaussian.com/gaussian16/) (a commercial quantum chemistry simulation software used for wavelength calculations)

# Run parallel MCTS algorithms for molecular design
## optimization of logP property
> mpiexec -n 4 python example_logp.py

## optimization of wavelength property
> mpiexec -n 4 python example_wavelength.py

> where 4 is the number of cores or processes to use. You can use more cores by changing 4 to 1024 for example. The example code used D-MCTS algorithm as default, you can simply change to H-MCTS for your own purpose by checking the source code of example_logp.py.

# Implement your own property simulator
> Go to pmcts folder and add your simulator to property_simulator.py
