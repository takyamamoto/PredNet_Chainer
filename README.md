# Deep Predictive Coding Network (PredNet) with Chainer
Deep Predictive Coding Network (**PredNet**) implemented with Chainer v4

## Requirement
- Python >=3.6
- Chainer 4.0
- matplotlib

## Usage

### Download Dataset
Download avi file from [First-Person Social Interactions Dataset](http://ai.stanford.edu/~alireza/Disney/).

### Train
When you use GPU, please run following command.
```
python train.py -g 0
```

## Results
Run following command.
```
python generate_result_images.py -g 0
```

<img src="https://github.com/takyamamoto/PredNet_Chainer/blob/master/results/out1.gif" width=70%>
<img src="https://github.com/takyamamoto/PredNet_Chainer/blob/master/results/out2.gif" width=70%>

## References
- [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
- [Unsupervised Learning of Visual Structure using Predictive Generative Networks](http://arxiv.org/abs/1511.06380)
- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
- [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
- [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
