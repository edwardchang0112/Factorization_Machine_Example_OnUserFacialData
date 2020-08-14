# Factorization_Machine_Example_OnUserFacialData
In this example, there is an use case for Factorization machine, which try to figure out what kind of user might be the target user (Frequent product use).

## Setting

1. $ sudo apt-get install python-dev libopenblas-dev

2. Clone the repo including submodules (or clone + `git submodule update --init --recursive`)

$ git clone --recursive https://github.com/ibayer/fastFM.git

3. Enter the root directory

$ cd fastFM

4. Install Python dependencies (Cython>=0.22, numpy, pandas, scipy, scikit-learn)

5. Compile the C extension.

$ make                      # build with default python version (python)

$ PYTHON=python3 make       # build with custom python version (python3)
6. Install fastFM

$ pip install .
7. Others requirements can be install in normal way.

This provides you an easy example, you can follow the structure, and replace with your data or make some modifications to fit your applications.
