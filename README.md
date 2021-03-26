## Audio Recognition Using CRNN, CTC Loss, DeepSpeech Beam Search and KenLM Scorer

[![medium](https://aleen42.github.io/badges/src/medium.svg)](https://dredwardhyde.medium.com/audio-recognition-using-crnn-ctc-loss-beam-search-decoder-and-kenlm-scorer-24472e43fb2f)
![Python3.8.6](https://img.shields.io/badge/Python-3.8.6-blue.svg)
![PyTorch1.8.1](https://img.shields.io/badge/PyTorch-1.8.1-yellow.svg)

### Architecture
```py
SpeechRecognitionModel(
  # First convolutional layer
  (cnn): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  # 5 blocks of convolutional layers with residual connections
  (res_cnn): Sequential(
    (0): ResidualCNN(
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ResidualCNN(
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ResidualCNN(
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (3): ResidualCNN(
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (4): ResidualCNN(
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (dropout): Dropout(p=0.2, inplace=False)
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  # Single linear layer to tie CNN and RNN parts together
  (fc): Linear(in_features=2048, out_features=512, bias=True)
  # 5 blocks of recurrent layers
  (rnn): Sequential(
    (0): RNN(
      (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (gru): GRU(512, 512, batch_first=True, bidirectional=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (1): RNN(
      (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (gru): GRU(1024, 512, bidirectional=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (2): RNN(
      (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (gru): GRU(1024, 512, bidirectional=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (3): RNN(
      (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (gru): GRU(1024, 512, bidirectional=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
    (4): RNN(
      (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU()
      (gru): GRU(1024, 512, bidirectional=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
  )
  # Two fully connected layers to convert the output of RNN layers to a probability 
  # distribution for each vertical feature vector and each character
  (dense): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=512, out_features=29, bias=True)
  )
)
```


### Training
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/training.png" width="773"/>  

### Results
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/results.png" width="1000"/>  

<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_1.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_2.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_3.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_4.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_5.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_6.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_7.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_8.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_9.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_10.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_11.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_12.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_13.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_14.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_15.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_16.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_17.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_18.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_19.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/audio-recognition/main/results/Figure_20.png" width="1000"/>  
