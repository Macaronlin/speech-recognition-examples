## Audio Recognition Using CRNN, CTC Loss, DeepSpeech Beam Search and KenLM Scorer

### Architecture
```py
SpeechRecognitionModel(
  (cnn): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
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
  (fc): Linear(in_features=2048, out_features=512, bias=True)
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
  (dense): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=512, out_features=29, bias=True)
  )
)
```


### Training
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/training.png" width="667"/>  

### Results
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/results.png" width="1000"/>  

<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_1.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_2.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_3.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_4.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_5.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_6.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_7.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_8.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_9.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_10.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_11.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_12.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_13.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_14.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_15.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_16.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_17.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_18.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_19.png" width="1000"/>  
<img src="https://raw.githubusercontent.com/dredwardhyde/librispeech-recognition/master/results/Figure_20.png" width="1000"/>  
