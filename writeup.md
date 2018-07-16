# **Traffic Sign Recognition** 

## Joe Cymerman
## 15 July 2018
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/joecym/P2_traffic_sign_classifier.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Here is the output from my script:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 1263
Image data shape = (32, 32, 3)
Number of classes = 42

#### 2. Include an exploratory visualization of the dataset.

 I used this script to load an image, and later made a loop to find an image of a particular type. This was helpful in ensuring that I correctly labeled the web images that I used in the last section. 
 
 ``` python
 
 index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

#plt.figure(figsize=(1,1))
#plt.imshow(image, cmap="gray")
#print(y_train[index])
count = 0


# I used this routine to verify that my images in the final part correctly matched the descriptions
while (count < 1000):
    index = random.randint(0, len(X_train))
    #print(y_train[index])
    if y_train[index] == 26:
        image = X_train[index].squeeze()
        plt.figure(figsize=(1,1))
        plt.imshow(image, cmap="gray")
        #print(y_train[index])
        break
    count = count + 1

 ```

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
I decided to convert the images to grayscale and normalize them as recommennded in the end. Initially I thought it would be more interesting to keep all three layers of color. For whatever reason I did not get better results with this method.  

You can open up my notebook and see a preview of a processed image. 

To make my model more accurate, I could have made additionall images by shifting them, but I did not quite get there. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers. It's very similar to the LeNET solition, but I increased the depths of the convolutional and fully connected layers to capture more details. 

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x30.
    # Activation.
    # Pooling. Input = 28x28x30. Output = 14x14x30.
    # Layer 2: Convolutional. Output = 10x10x40.
    # Activation.
    # Pooling. Input = 10x10x40. Output = 5x5x40.
    # Flatten. Input = 5x5x16. Output = 1000.
    # Layer 3: Fully Connected. Input = 1000. Output = 300.
    # Activation.
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Activation.

# Layer 5: Fully Connected. Input = 84. Output = 10.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 
To train the model, I used a learning rate of 0.002, 5 epochs, and a batch size of 128.

#### 4. I changed the depths of the layers and experimented with pre-processing of the images to achieve a validation accuracy of at least 0.93. 


My final model results were



* validation set accuracy of 94.8%
* test set accuracy of 93.7%


I believed a well known architecture like LeNet would successfully solve this problem. However, I was not able to achieve the same level  of accuracy as the simpler LeNet solution. I had problems with overfitting, as my accuracy would decline rapidly after a certain number of epochs. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose five images on the web and you can see them printed in my notebook. The "children crossing" image was the most difficult because of the angle and the resolution of the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			        |     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		| 50 km/h   									| 
| Bumpy Road     			| Bumpy Road 										|
| No Entry				| No Entry											|
| Children Crossing	      		| Traffic Signals					 				|
| Roundabout		| Right of Way      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is less than the test set but I think it may be due to the poor quality of the images after I had resized it to fit the 32x32 shape, which affected my last two images the most.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is almost certain that the 50 km/hr sign is a 50 km/hr sign. The top five soft max probabilities were

| Probability         	|     Prediction	  		| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 50 km/hr 									| 
|  3.76e-4    				|30 km/hr										|
| 1.15e-4					| 80 km/hr										|
| 1.04e-5	      			| Wild Animals				 				|
| 5.18e-7			    | Stop      							|

For the second image, the model is virtually certain that the bumpy road sign is a bumpy road sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Bumpy Road   									| 
|  7.06e-4    				|Bike Crossing										|
| 3.92e-4					| No Vehicles											|
| 7.30e-5	      			| Road Work				 				|
| 3.92e-7			    | Priority Road      							| 

For the third image, the model is mostly certain that the No Entry image is a No Entry.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| No Entry   									| |  1.8e-4    				|Turn Left Ahead										|
| 2.63e-9					| Ahead Only											|
| 4.5e-10	      			| Go Straight or Left			 				|
| 7.08e-11			    | Priority Road      							|

For the fourth image, the model is mostly certain that the children crossing sign is a Traffic Signals Sign, which is incorrect.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71        			| Traffic Signals  									| 
|  .19   				|Turn Right Ahead										|
| .09					| Dangerous Curve Right											|
| .003	      			| Keep Left		 				|
| .0017			    | Go Straight or Left     							|

For the fifth image, the model is almost certain that the roundabout sign is a Right of Way Sign, which is incorrect.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999        			| Right of Way 									| 
| 9.68e-4   				|Double Curve										|
| 2.93e-4					| Ice											|
| 2.78e-5	      			| Roundabout 				|
| 6.5e-7			    | Go Straight or Left     							|
