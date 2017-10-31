#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/images/image1.jpg "Traffic Sign 1"
[image5]: ./examples/images/image2.jpg "Traffic Sign 2"
[image6]: ./examples/images/image3.jpg "Traffic Sign 3"
[image7]: ./examples/images/image4.jpg "Traffic Sign 4"
[image8]: ./examples/images/image5.jpg "Traffic Sign 5"
[image9]: ./examples/images/image6.jpg "Traffic Sign 6"
[image10]: ./examples/predicion.png "Prediction Results"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Xfan1025/SDCND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the sign classes/labels are distributed. After visualizing the distributions in each dataset, I conclude the distribution is roughly the same but classes of 'Speed Limit of 50km', 'Speed Limit of 30km' have more examples than others. This may result in better predictions on Speed limit signs.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because the grayscale images are simpler than RGB images. The model should have an easier time to learn grayscale images than RGB images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data using min-max scaling as feature normalization is a common and powerful technique to prevent data getting too large or too small during computation. It will improve the accuracy and also improve the training speed.




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Leaky_RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Leaky_RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x86	|
| Leaky_RELU					|												|
| Flatten		| output 3*3*86=774        									|
| Fully connected      | Leaky_RELU, dropout, outputs 256|
| Fully connected      | Leaky_RELU, outputs 64|
| Fully connected      | Leaky_RELU, outputs 43|
| Softmax				|        		final output							|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer....

| Parameters         		|     Value	        					|
|:---------------------:|:---------------------------------------------:|
| Optimizer         		| Adam  							|
| Loss   	| Cross entropy loss 	|
| Batch size					|						128						|
| Number of epochs	      	| 35 				|
| Learning Rate    | 0.001	|

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 0.997
* validation set accuracy of ? 0.948
* test set accuracy of ? 0.924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first tried 2 conv layers and used with RGB images since the LeNet architecture looks great fit to RGB images, and the validation accuracy was around 0.90.
* What were some problems with the initial architecture?
The accuracy stopped at 0.91 and couldn't go better.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added one conv layer hoping the model can learn smaller details of the images. And I also applied dropout to first FC layer to prevent overfitting.
* Which parameters were tuned? How were they adjusted and why?
I started with learning rate of 0.0001 for the new architecture. The accuracy increased slowly. After some playing around, I found 0.001 is quite reasonable.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layer shares parameters across kernels which can help us to recognize the objects regardless their locations in the image. This is why ConvNet is so powerful for image classifications task. The dropout layer can improve the training speed as it reduced the total number of pixels needed for calculation. And because we intentionally drop out some pixels randomly, the model will become more robust in order to accept those data.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]


The last image might be difficult to classify because it got too much noise after preprocessing.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right ahead      		| Turn right ahead   									|
| Stop     			| Stop 										|
| Speed limit (60km/h)					| Speed limit (60km/h)											|
| Speed limit (70km/h)	      		| Speed limit (70km/h)					 				|
| Ahead only		| Ahead only      							|
| Traffic signals		| Go straight or left      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the last image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .891         			| Go straight or left   									|
| .094     				| Keep right 										|
| .013					| Roundabout mandatory											|
| .002	      			| General caution				 				|
| .000				    | Traffic signals      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
