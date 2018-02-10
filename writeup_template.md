**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* Google drive link for [model.h5](https://drive.google.com/file/d/1lkTPwXSUXxYkGdmPDBWnBDOP-DErxndA/view?usp=sharing) containing a trained convolution neural network
* writeup_template.md for summarizing the results
* [Youtube link](https://youtu.be/c3pN45nM5yY) for the final project output

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is basically inspired from the [NVidia architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) along with some modification of my own which is clearly stated in the `build_model()` function of model.py python file.

The model includes ELU(Exponential Linear Units) which is better than the sigmoid and RELU activation units which all basically adds non-linearity to the output and make it differentiable to be used in gradient descent. For more details, [Read this blog](http://saikatbasak.in/sigmoid-vs-relu-vs-elu/). I have also added normalization layer which is basically kera lamda layer and cropping layer to the model as the top 70 pixels and bottom 25 pixels of the image, I found out not to be useful in training by experimentation.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I basically started from the training data given by Udacity and added a combination of recovery lapse from the sides, laps focusing on driving smoothly around curves and counter clockwise driving to reduce the left turn bias and generalize better. I didn't used the data from the second track as the data from the first track was sufficient enough for the problem in hand.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia Architecture, I thought this model might be appropriate because it's a powerful network and has shown promising results. Even it has been proposed in the classroom lectures as well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model and added the dropout layers so the overfitting was corrected but still the car was sometimes going off track mostly during turns and was having left turn bias.

Then I preprocessed the image to add random flipping and also collected the data for counter clockwise driving, mostly to avoid left turn bias. For going offtrack problem, first I collected recovery lapse data i.e I recorded data when the car is driving from the sides of the road back towards the center line and then I added laps mainly focussing on driving smoothly around curves. I also added a data augmentation step to reduce the brightness of the images, as I found out some images are very bright than others and it worked well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

| Layer (type)         | Output Shape   | Param |
|:-------------:|:-------------:|
| lambda_1 (Lambda)      | (None, 160, 320, 3)        | 0 |
| cropping2d_1 (Cropping2D)      | (None, 65, 320, 3)      | 0 |
| conv2d_1 (Conv2D)     | (None, 33, 160, 32)      | 2432 |
| elu_1 (ELU)      | (None, 33, 160, 32)       | 0 |
| conv2d_2 (Conv2D)      | (None, 31, 158, 16)       | 4624 |
| elu_2 (ELU)      | (None, 31, 158, 16)       | 0 |
| dropout_1 (Dropout)      | (None, 31, 158, 16)      | 0 |
| max_pooling2d_1 (MaxPooling2D)      | (None, 15, 79, 16)      | 0 |
| conv2d_3 (Conv2D)      | (None, 13, 77, 16)       | 2320 |
| elu_3 (ELU)      | (None, 13, 77, 16)       | 0 |
| dropout_2 (Dropout)      | (None, 13, 77, 16)         | 0 |
| flatten_1 (Flatten)      | (None, 16016)       | 0 |
| dense_1 (Dense)      | (None, 1024)        | 16401408 |
| dropout_3 (Dropout)      | (None, 1024)       | 0 |
| elu_4 (ELU)      | (None, 1024)       | 0 |
| dense_2 (Dense)      | (None, 512)       | 524800 |
| elu_5 (ELU))      | (None, 512)       | 0 |
| dense_3 (Dense)      | (None, 1)       | 513 |

* Total params: 16,936,097
* Trainable params: 16,936,097
* Non-trainable params: 0
_________________________________________________________________
Train on 11049 samples, validate on 2763 samples



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
