This is the code base for paper [Enhancing Space-time Video Super-resolution via Spatial-temporal Feature Interaction](https://arxiv.org/abs/2207.08960)

# Prepare data
  Please download the dataset, i.e., [Vimeo-90K Septuplet Dataset](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip), and put it under datasets/
# Install and compile the prerequisites
- Python 3.6
- PyTorch >= 1.1
- NVIDIA GPU + CUDA
- Deformable Convolution v2 (DCNv2), we adopt [CharlesShang's implementation](https://github.com/CharlesShang/DCNv2). (see and compile DCNv2/make.sh) 
- Python packages: numpy,opencv-python,dgl
# Pretrained model
Please download the [pretrained model](https://drive.google.com/file/d/1fYM_PMQof-XFRFaw9LV2z7MI4X4L22al/view?usp=sharing), and put it under weights/pretrained/

# Main experiment

see [python main_STINet.py]


# Citation
```
@article{yue2022enhancing,
  title={Enhancing Space-time Video Super-resolution via Spatial-temporal Feature Interaction},
  author={Yue, Zijie and Shi, Miaojing and Ding, Shuai and Yang, Shanlin},
  journal={arXiv preprint arXiv:2207.08960},
  year={2022}
}
```
