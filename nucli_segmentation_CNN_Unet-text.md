
# 2018 Data Science Bowl project
## Find the nuclei in divergent images to advance medical discovery
* https://www.kaggle.com/c/data-science-bowl-2018
* Step 1. load and preprocess the data
* Step 2. visualization to to plan normalization scheme
* Step 3. normalization of the data
* Step 4. train data generator and data augmentation
* Step 5. build and train a simple CNN
* Step 6. build and train Unet CNN
* Step 7.  Unet model on the test data
* Step 8. apply and compare Simple CNN with Unet results
* source: https://www.kaggle.com/kmader/nuclei-overview-to-submission

In this project, I investigate the potential 2D U-net to segment the nuclei in divergent images. Deep learning U-net neural network, has proven lately as efficient tool for many biomedical segmentation tasks. The nature of most medical images, that have high correlation among near voxel, make them good candidate for analyzing with convolution sliding window.


## Step 1. Load the data
View aggregation on images parameters, such as sizes, number of masks and channles.


```python

```

    train_df:
            img_height    img_width   img_ratio  num_channels   num_masks
    count   670.000000   670.000000  670.000000         670.0  670.000000
    mean    333.991045   378.500000    0.658209           3.0   43.971642
    std     149.474845   204.838693    0.474664           0.0   47.962530
    min     256.000000   256.000000    0.000000           3.0    1.000000
    25%     256.000000   256.000000    0.000000           3.0   15.250000
    50%     256.000000   320.000000    1.000000           3.0   27.000000
    75%     360.000000   360.000000    1.000000           3.0   54.000000
    max    1040.000000  1388.000000    1.000000           3.0  375.000000


We can see that all images has 3 channles, also images differ in sizes.<br>
Next Read images/masks from files
and stored them as list of a 3-dim array where the number of channels is 3 for images 1 for masks.<br>


## Step 2. Data visualization
Here we show a few images of the cells where we see there is a mixture of brightfield and fluorescence which will probably make using a single segmentation algorithm difficult



```python
show_img_mask_grid()
```


![png](output_6_0.png)


## Step 3. Normalize
We can see that images differ in intenity scale. To normalize, we will invert the bright images, and normalize all to [0,1]. After testing few normalization techniques the following were chosen:<br>
I tried  histogram equalizer (that a lot of resources and did not improve the results), STD did not improved results as well.


```python
def normalize(x):
    for i in range(x.shape[2]):
        div = np.max(x[:,:,i]) 
        if (div < 0.01*x[:,:,i].mean()): div = 1. # protect against too small pixel intensities
        x[:,:,i] = x[:,:,i].astype(np.float32)/div
    return x
```


```python
def invert(x, cutoff=0.5):
    
    print(x.min(), x.max(), x.mean())
    if (np.mean(x)>cutoff):
        x = 1. - x
    return x
```

Lets view some normalized images and masks:


```python
show_img_mask_grid(row=2, col=4, im=x_train, msk=x_train_norm, label_msk = 'normalized')
```


![png](output_11_0.png)


## Step 4. train data generator and data augmentation

### Check Dimensions 
Here we show the dimensions of the data to see the variety in the input images


```python
train_df['shapes'] = train_df.apply(lambda row: np.array(read_image(row['image_path'])).shape, axis=1)
```


```python
train_df['shapes'].value_counts()
```




    (256, 256, 3)      334
    (256, 320, 3)      112
    (520, 696, 3)       92
    (360, 360, 3)       91
    (1024, 1024, 3)     16
    (512, 640, 3)       13
    (603, 1272, 3)       6
    (260, 347, 3)        5
    (1040, 1388, 3)      1
    Name: shapes, dtype: int64



### Data augmentation
The simple  CNN will be for any gien size. We will train it with 256x256 croped images, and we apply it to predict any image size. Data augmentation is simple, a randomly chosen rotation and flip, and then a random patch is cut out of the image in the size of 256x256.

### Split the data to Test and Train


```python
from sklearn.model_selection import train_test_split
x_trn, x_vld, y_trn, y_vld = train_test_split( x_train, y_train, test_size=0.1, random_state=42)
```

### Generate train and validation data
The generator will generate batches which are numpy arrays of size 256x256. For train data we will use augmentation, and for validation data only crop.

Following is a plot of augmented images.


```python
show_img_grid(row=2, col=4, im=x)
```


![png](output_21_0.png)


## Step 5. Build a simple CNN
Here we make a very simple CNN just for a benchmark. For this we use a batch normalization to normalize the inputs. We cheat a bit with the padding to keep problems simple.


```python
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda
simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape = (None, None, IMG_CHANNELS), 
                                  name = 'NormalizeInput'))
simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
# use dilations to get a slightly larger field of view
simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))

# the final processing
simple_cnn.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))
simple_cnn.add(Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))
```


```python
simple_cnn.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    NormalizeInput (BatchNormali (None, None, None, 3)     12        
    _________________________________________________________________
    conv2d_60 (Conv2D)           (None, None, None, 8)     224       
    _________________________________________________________________
    conv2d_61 (Conv2D)           (None, None, None, 8)     584       
    _________________________________________________________________
    conv2d_62 (Conv2D)           (None, None, None, 16)    1168      
    _________________________________________________________________
    conv2d_63 (Conv2D)           (None, None, None, 16)    2320      
    _________________________________________________________________
    conv2d_64 (Conv2D)           (None, None, None, 32)    4640      
    _________________________________________________________________
    conv2d_65 (Conv2D)           (None, None, None, 16)    528       
    _________________________________________________________________
    conv2d_66 (Conv2D)           (None, None, None, 1)     17        
    =================================================================
    Total params: 9,493
    Trainable params: 9,487
    Non-trainable params: 6
    _________________________________________________________________


# Loss
Since we are being evaulated with intersection over union we can use the inverse of the DICE score as the loss function to optimize


```python
from keras import backend as K
smooth = 1.
def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

```


```python
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

```


```python
from keras.optimizers import Adam
learning_rate=0.001
simple_cnn.compile(optimizer=Adam(lr=learning_rate),
                           loss = dice_coef_loss, 
                   metrics = ['binary_crossentropy', jaccard_distance_loss])
```

# Simple Training
Train the model for 6 epoch, with generated train data.


```python
plot_history(history, 'loss')
```


![png](output_30_0.png)



```python
plot_history(history, key='jaccard_distance_loss')
```


![png](output_31_0.png)



```python
plot_history(history, key='binary_crossentropy')
```


![png](output_32_0.png)


## Unet
### U-Net: Convolutional Networks
U-Net is a very popular end-to-end encoder-decoder network for semantic segmentation [7]. It's was originally invented and first used for biomedical image segmentation. U-net’s have multi channel architecture, which suits well the nuclei in divergent images.<br>
Medical MRI images feature a high similarity and correlation in the intensities among neighboring voxels, so they are good candidates for the convolution blocks constructing the U-net.<br>
### Model Architecture
 The architecture we used, had 3 channels for the input, and 1 channels in the output. We used a 2D architecture.
 
### U-Net: Model Architecture
U-Net is a very popular end-to-end encoder-decoder network for semantic segmentation. It was originally invented and first used for biomedical image segmentation. U-net’s have multi channel architecture, which suits well the multi channel input of MRI images and the multi-class classification task.<br>
Medical images feature a high similarity and correlation in the intensities among neighboring voxels, so they are good candidates for the convolution blocks constructing the U-net.<br>
Essentially, U-net is a deep-learning framework based on fully convolutional networks, it comprises two parts:<br>
A contracting path similar to an encoder, to capture context from a compact feature representation.<br>
A symmetric expanding path similar to a decoder, which allows accurate  localisation. This step is done to retain boundary information (spatial information) despite down sampling and max-pooling performed in the encoder stage.<br>
Instead of up-sampling methods,in U-net, we concatenate the suitable transformed layer from the decoder path.<br>
** image**


```python
from keras import backend as K
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam

def get_unet(IMG_HEIGHT=None, IMG_WIDTH=None, IMG_CHANNELS=3, n_ch_output=1):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(n_ch_output, (1, 1), activation='sigmoid', padding='same') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
```

Train Unet model for 10 epoch, with train data generator.


```python
unet.save_weights('unet_first10epoch_dc_0.8867.hdf5')
```


```python
plot_history(history_unet, 'loss')
```


![png](output_37_0.png)



```python
plot_history(history_unet, 'binary_crossentropy')
```


![png](output_38_0.png)



```python
plot_history(history_unet, 'jaccard_distance_loss')
```


![png](output_39_0.png)


Train another 2 epoch with jaccard_distance_loss, improve results.


```python
history_unet2 = unet.fit_generator(train_gen, steps_per_epoch=len(x_train)//10,
                        validation_data = (x_valid, y_valid), epochs = 2,
                                  callbacks=[early_stopping, checkpointer])
```

    Epoch 1/2
    67/67 [==============================] - 125s 2s/step - loss: 0.0498 - binary_crossentropy: 0.5469 - dice_coef_loss: -0.8069 - val_loss: 0.0293 - val_binary_crossentropy: 0.2758 - val_dice_coef_loss: -0.8743
    
    Epoch 00001: val_loss improved from inf to 0.02930, saving model to model_unet.01-0.029303.hdf5
    Epoch 2/2
    67/67 [==============================] - 130s 2s/step - loss: 0.0544 - binary_crossentropy: 0.6017 - dice_coef_loss: -0.8101 - val_loss: 0.0267 - val_binary_crossentropy: 0.2244 - val_dice_coef_loss: -0.8905
    
    Epoch 00002: val_loss improved from 0.02930 to 0.02673, saving model to model_unet.02-0.026732.hdf5


## Step 8. apply and compare Simple CNN with Unet results

### Build validation data for predicting
U-net architecrure, requires that image and label sizes will be devided by 128. we pad the images and masks with reflection, to the nearest size which is devidable by 128.

View few validation images and masks


```python
show_img_mask_grid(row=2, col=4, im=x_for_pred, 
                   msk=[y.reshape(y.shape[0], y.shape[1]) for y in y_for_pred])
```


![png](output_45_0.png)



```python
y_test_pred_proba_cnn = simple_cnn.predict_proba(np.array(x_for_pred[0:2]))
```


```python
y_test_pred_proba_unet = unet.predict(np.array(x_for_pred[0:2]))
```


```python
show_img_inline([x_for_pred[0], y_test_pred_proba_unet[0,:,:,0], 
                 y_test_pred_proba_cnn[0,:,:,0],
                 y_for_pred[0][:,:,0]],
                labels=['img', 'Unet pred', 
                        'pred_cnn','true label'])
```


![png](output_48_0.png)



```python
show_img_inline([x_for_pred[1], y_test_pred_proba_unet[1,:,:,0], 
                 y_test_pred_proba_cnn[1,:,:,0],
                 y_for_pred[1][:,:,0]],
                labels=['img', 'Unet pred', 
                        'pred_cnn','true label'])
```


![png](output_49_0.png)


## Results
The performance of U-net’s was mixed for this segmentation task. 
* The U-net perform well, when segmentation had “blobe” structure, rather then sparse structure.

* The U-net did not separate well between label 4 and 1, and had bad performance on sparse pixel segmentation.

> Simple CNN results: loss: -0.7876 - binary_crossentropy: 0.7696 - jaccard_distance_loss: 0.0611 - val_loss: -0.8327 - val_binary_crossentropy: 0.4176 - val_jaccard_distance_loss: 0.0395

> Unet resolts: loss: -0.8122 - binary_crossentropy: 0.5806 - jaccard_distance_loss: 0.0533 - val_loss: -0.8867 - val_binary_crossentropy: 0.2350 - val_jaccard_distance_loss: 0.0282

# Discussion
U-nets model yield good result in segmenting nuclei in divergent images. To improve the model, I suggest to explore following paths:  
* Preprocessing the data 
Images in data set, are highly variant in sizes and color range, specific equalization technicks, that are suitable to medical imaging should be explored.
* Resampling techniques
In resanpling tecniques the meaning is how we design the train image generator.
* Optional loss functions
The most popular measure in segmentation is IOU. IOU is not defrentable, so it can not be used in backpropogation. Dice Coefficient, is design to give close results to IOU. 
* Data augmentation
Data augmentation, improve results signicantly. The data in this project is highly suitable for augmentation, since to begin with, there is no meanning to orientation and resizing.
* Network architecture
The U-net was not sensitive enough, different architectures, that encodes to smaller size can improve the sensitivity of the segmentation.
* Training parameters
Parameters as batch size, learning rate, optimizers, should be explored with better hardware then I have.<br>

Most promising direction, is a combination of Unet with LSTM that will "sweep" the image to keep track on the location of the specific nucli.

## Articles:
* Abdel Aziz, Allan Hanbury. **Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool**. BMC Med Imaging. 2015; 15: 29. 
* Ilya E Vorontsov†, Ivan V Kulakovskiy, Vsevolod J Makeev. **Jaccard index based similarity measure to compare transcription factor binding site models**. Algorithms for Molecular Biology Sep 2013.
* 1L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L.Yuille. **Semantic image segmentation with deep convolutional nets and fully connected crfs.** In ICLR, 2015.
* Havaei, M. et. al, Brain Tumor Segmentation with Deep Neural Networks. arXiv preprint arXiv:1505.03540, 2015.Author, F., Author, S., Author, T.: Book title. 2nd edn. Publisher, Location (1999).
* Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: **Convolutional Networks for Biomedical Image Segmentation.** Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at arXiv:1505.04597 [cs.CV]
* Sørensen, T. (1948). **"A method of establishing groups of equal amplitude in plant sociology based on similarity of species and its application to analyses of the vegetation on Danish commons".** Kongelige Danske Videnskabernes Selskab. 5 (4): 1–34.
* J.G. Sled, A.P. Zijdenbos and A.C. Evans. "A Nonparametric Method for Automatic Correction of Intensity Nonuniformity in Data" IEEE Transactions on Medical Imaging, Vol 17, No 1. Feb 1998.
* Chengjia Wang, Tom MacGillivray, Gillian Macnaught, Guang Yang, David Newby. **“A two-stage 3D Unet framework for multi-class segmentation on full resolution image”.**  2016. arXiv:1606.06650 [cs.CV]
* Sadanandan, Sajith Kecheril, Ranefall, Petter, Sotiras A, Bilello M, Le Guyader, Sylvie, Wählby, Carolina. **"Automated Training of Deep Convolutional Neural Networks for Cell Segmentation"**, Scientific Reports,2045-2322 (2017) DOI: https://doi.org/10.1038/s41598-017-07599-6
* Ioffe, S. & Szegedy, C. **Batch normalization: Accelerating deep network training by reducing internal covariate shift.** In Proceedings of The 32nd International Conference on Machine Learning 448–456 (2015).
* He, K., Zhang, X., Ren, S. & Sun, J. **Delving deep into rectifiers: Surpassing human-level performance on imagenet classification**. In Proceedings of the IEEE international conference on computer vision 1026–1034 (2015).
* Nair, V. & Hinton, G. E. **Rectified linear units improve restricted boltzmann machines.** In Proceedings of the 27th international conference on machine learning (ICML-10) 807–814 (2010).
* 1L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L.Yuille. **Semantic image segmentation with deep convolutional nets and fully connected crfs**. In ICLR, 2015. https://arxiv.org/abs/1412.7062


Keras: The Python Deep Learning library, https://keras.io/

