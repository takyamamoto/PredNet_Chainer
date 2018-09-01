# Deep Predictive Coding Network (PredNet) with Chainer
Deep Predictive Coding Network ([PredNet](https://github.com/coxlab/prednet)) implemented with Chainer v4

## Requirement
You must use a GPU to train.
- Python >=3.6
- Chainer 4.0
- opencv-python
- matplotlib
- tqdm
- glob
- shutil

## Usage

### Download Dataset
Download preprocessed KITTI datasets from [prednet_kitti_data.zip](https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0 -O $savedir/prednet_kitti_data.zip).  
You have to convert hickle-python2 data to npy data.

### Train
When you use GPU, please run following command.
```
python train.py -g 0
```

When you want to train extrapolation model, run
```
python train_extrap.py -g 0
```

## Results
Run following command.
```
python generate_result_images.py -g 0
```
or
```
python generate_result_images_extrap.py -g 0
```

<img src="https://github.com/takyamamoto/PredNet_Chainer/blob/master/results/output.gif" width=70%>
<img src="https://github.com/takyamamoto/PredNet_Chainer/blob/master/results/output2.gif" width=70%>

## References
- [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
- [Unsupervised Learning of Visual Structure using Predictive Generative Networks](http://arxiv.org/abs/1511.06380)
- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
- [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
- [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
