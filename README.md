# Prepare data
  Please download the dataset, i.e., [Vimeo-90K Septuplet Dataset](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip), and put it under datasets/
# Install and compile the prerequisites
- Python 3.6
- PyTorch >= 1.1
- NVIDIA GPU + CUDA
- Deformable Convolution v2 (DCNv2), we adopt [CharlesShang's implementation](https://github.com/CharlesShang/DCNv2). (see and compile DCNv2/make.sh) 
- Python packages: numpy,opencv-python,dgl

# Main experiment

see [python main_STINet.py]