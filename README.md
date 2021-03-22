## LibriSpeech Dataset Audio Recognition Using CRNN, CTC Loss, DeepSpeech Beam Search and KenLM Scorer

### Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 32, 64, 672]             320
         LayerNorm-2          [-1, 32, 672, 64]             128
              GELU-3          [-1, 32, 64, 672]               0
           Dropout-4          [-1, 32, 64, 672]               0
            Conv2d-5          [-1, 32, 64, 672]           9,248
               CNN-6          [-1, 32, 64, 672]               0
         LayerNorm-7          [-1, 32, 672, 64]             128
              GELU-8          [-1, 32, 64, 672]               0
           Dropout-9          [-1, 32, 64, 672]               0
           Conv2d-10          [-1, 32, 64, 672]           9,248
              CNN-11          [-1, 32, 64, 672]               0
      ResidualCNN-12          [-1, 32, 64, 672]               0
        LayerNorm-13          [-1, 32, 672, 64]             128
             GELU-14          [-1, 32, 64, 672]               0
          Dropout-15          [-1, 32, 64, 672]               0
           Conv2d-16          [-1, 32, 64, 672]           9,248
              CNN-17          [-1, 32, 64, 672]               0
        LayerNorm-18          [-1, 32, 672, 64]             128
             GELU-19          [-1, 32, 64, 672]               0
          Dropout-20          [-1, 32, 64, 672]               0
           Conv2d-21          [-1, 32, 64, 672]           9,248
              CNN-22          [-1, 32, 64, 672]               0
      ResidualCNN-23          [-1, 32, 64, 672]               0
        LayerNorm-24          [-1, 32, 672, 64]             128
             GELU-25          [-1, 32, 64, 672]               0
          Dropout-26          [-1, 32, 64, 672]               0
           Conv2d-27          [-1, 32, 64, 672]           9,248
              CNN-28          [-1, 32, 64, 672]               0
        LayerNorm-29          [-1, 32, 672, 64]             128
             GELU-30          [-1, 32, 64, 672]               0
          Dropout-31          [-1, 32, 64, 672]               0
           Conv2d-32          [-1, 32, 64, 672]           9,248
              CNN-33          [-1, 32, 64, 672]               0
      ResidualCNN-34          [-1, 32, 64, 672]               0
           Linear-35             [-1, 672, 512]       1,049,088
        LayerNorm-36             [-1, 672, 512]           1,024
             GELU-37             [-1, 672, 512]               0
              GRU-38           [[-1, 672, 1024], 
                                   [-1, 2, 512]]              0
          Dropout-39            [-1, 672, 1024]               0
              RNN-40            [-1, 672, 1024]               0
        LayerNorm-41            [-1, 672, 1024]           2,048
             GELU-42            [-1, 672, 1024]               0
              GRU-43           [[-1, 672, 1024],
                                 [-1, 672, 512]]              0
          Dropout-44            [-1, 672, 1024]               0
              RNN-45            [-1, 672, 1024]               0
        LayerNorm-46            [-1, 672, 1024]           2,048
             GELU-47            [-1, 672, 1024]               0
              GRU-48           [[-1, 672, 1024], 
                                 [-1, 672, 512]]              0
          Dropout-49            [-1, 672, 1024]               0
              RNN-50            [-1, 672, 1024]               0
        LayerNorm-51            [-1, 672, 1024]           2,048
             GELU-52            [-1, 672, 1024]               0
              GRU-53           [[-1, 672, 1024], 
                                 [-1, 672, 512]]              0
          Dropout-54            [-1, 672, 1024]               0
              RNN-55            [-1, 672, 1024]               0
        LayerNorm-56            [-1, 672, 1024]           2,048
             GELU-57            [-1, 672, 1024]               0
              GRU-58           [[-1, 672, 1024], 
                                 [-1, 672, 512]]              0
          Dropout-59            [-1, 672, 1024]               0
              RNN-60            [-1, 672, 1024]               0
           Linear-61             [-1, 672, 512]         524,800
             GELU-62             [-1, 672, 512]               0
          Dropout-63             [-1, 672, 512]               0
           Linear-64              [-1, 672, 29]          14,877
================================================================
Total params: 1,654,557
Trainable params: 1,654,557
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.66
Forward/backward pass size (MB): 11475.40
Params size (MB): 6.31
Estimated Total Size (MB): 11482.37
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