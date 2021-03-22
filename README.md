## LibriSpeech Dataset Audio Recognition Using CRNN, CTC Loss, DeepSpeech Beam Search and KenLM Scorer

### Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 32, 64, 672]             320
         LayerNorm-2          [-1, 32, 672, 64]             128
      CNNLayerNorm-3          [-1, 32, 64, 672]               0
           Dropout-4          [-1, 32, 64, 672]               0
            Conv2d-5          [-1, 32, 64, 672]           9,248
         LayerNorm-6          [-1, 32, 672, 64]             128
      CNNLayerNorm-7          [-1, 32, 64, 672]               0
           Dropout-8          [-1, 32, 64, 672]               0
            Conv2d-9          [-1, 32, 64, 672]           9,248
      ResidualCNN-10          [-1, 32, 64, 672]               0
        LayerNorm-11          [-1, 32, 672, 64]             128
     CNNLayerNorm-12          [-1, 32, 64, 672]               0
          Dropout-13          [-1, 32, 64, 672]               0
           Conv2d-14          [-1, 32, 64, 672]           9,248
        LayerNorm-15          [-1, 32, 672, 64]             128
     CNNLayerNorm-16          [-1, 32, 64, 672]               0
          Dropout-17          [-1, 32, 64, 672]               0
           Conv2d-18          [-1, 32, 64, 672]           9,248
      ResidualCNN-19          [-1, 32, 64, 672]               0
        LayerNorm-20          [-1, 32, 672, 64]             128
     CNNLayerNorm-21          [-1, 32, 64, 672]               0
          Dropout-22          [-1, 32, 64, 672]               0
           Conv2d-23          [-1, 32, 64, 672]           9,248
        LayerNorm-24          [-1, 32, 672, 64]             128
     CNNLayerNorm-25          [-1, 32, 64, 672]               0
          Dropout-26          [-1, 32, 64, 672]               0
           Conv2d-27          [-1, 32, 64, 672]           9,248
      ResidualCNN-28          [-1, 32, 64, 672]               0
           Linear-29             [-1, 672, 512]       1,049,088
        LayerNorm-30             [-1, 672, 512]           1,024
              GRU-31  [[-1, 672, 1024], [-1, 2, 512]]         0
          Dropout-32            [-1, 672, 1024]               0
 BidirectionalGRU-33            [-1, 672, 1024]               0
        LayerNorm-34            [-1, 672, 1024]           2,048
              GRU-35  [[-1, 672, 1024], [-1, 672, 512]]       0
          Dropout-36            [-1, 672, 1024]               0
 BidirectionalGRU-37            [-1, 672, 1024]               0
        LayerNorm-38            [-1, 672, 1024]           2,048
              GRU-39  [[-1, 672, 1024], [-1, 672, 512]]       0
          Dropout-40            [-1, 672, 1024]               0
 BidirectionalGRU-41            [-1, 672, 1024]               0
        LayerNorm-42            [-1, 672, 1024]           2,048
              GRU-43  [[-1, 672, 1024], [-1, 672, 512]]       0
          Dropout-44            [-1, 672, 1024]               0
 BidirectionalGRU-45            [-1, 672, 1024]               0
        LayerNorm-46            [-1, 672, 1024]           2,048
              GRU-47  [[-1, 672, 1024], [-1, 672, 512]]       0
          Dropout-48            [-1, 672, 1024]               0
 BidirectionalGRU-49            [-1, 672, 1024]               0
           Linear-50             [-1, 672, 512]         524,800
             GELU-51             [-1, 672, 512]               0
          Dropout-52             [-1, 672, 512]               0
           Linear-53              [-1, 672, 29]          14,877
================================================================
Total params: 1,654,557
Trainable params: 1,654,557
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.66
Forward/backward pass size (MB): 11388.77
Params size (MB): 6.31
Estimated Total Size (MB): 11395.74
----------------------------------------------------------------
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