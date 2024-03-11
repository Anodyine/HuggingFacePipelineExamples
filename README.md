## Requirements
Nvidia Graphics Card with updated drivers

## Installation Steps
### Install Python 3.12 

You can download installers from here:

https://www.python.org/downloads/

Choose install pip and add to environment variables during installation

### Install PyTorch with Cuda
In console, type:

nvidia-smi

If it doesn't work, you may need to update the drivers.

Note the CUDA Version in the top right.

Go here:

https://pytorch.org/get-started/locally/

Choose your OS, Pip, Python, and your CUDA Version

Run the command given there with pip instead of pip3

For example this is the command that works for me:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### Install Python Packages
Run the following command:

pip install transformers pandas matplotlib accelerate scikit-learn seaborn

