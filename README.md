# Final project for Deep Learning section in Udacity course "Introduction to Machine Learning with TensorFlow Nanodegree Program"



## Part 1: Developing an Image Classifier with Deep Learning using TensorFlow
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories, you can see a few examples below. 

![Image of Samples](https://github.com/namnhatpham1995/Udacity-Image-Classifier-TensorFlow/blob/master/test_images/Flowers.png)

The project is broken down into multiple steps:

* Load the image dataset and create a pipeline.
* Build and Train an image classifier on this dataset.
* Use your trained model to perform inference on flower images.

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

## Part 2: Building the Command Line Application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a Python script that run from the command line. For testing, you should use the saved Keras model you saved in the first part.

Specifications
The project submission must include a predict.py file that uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a module just for utility functions like preprocessing images. Make sure to include all files necessary to run the predict.py file in your submission.

The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

Basic usage:
,,,
$ python predict.py /path/to/image saved_model
,,,
Options:

--top_k : Return the top KK most likely classes:
$ python predict.py /path/to/image saved_model --top_k KK
--category_names : Path to a JSON file mapping labels to flower names:
$ python predict.py /path/to/image saved_model --category_names map.json
The best way to get the command line input into the scripts is with the argparse module in the standard library. You can also find a nice tutorial for argparse here.

Examples
For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

Basic usage:

$ python predict.py ./test_images/orchid.jpg my_model.h5
Options:

Return the top 3 most likely classes:
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
Use a label_map.json file to map labels to flower names:
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
Workspace
Install TensorFlow
We have provided a Command Line Interface workspace for you to run and test your code. Before you run any commands in the terminal make sure to install TensorFlow 2.0 and TensorFlow Hub using pip as shown below:

$ pip install -q -U "tensorflow-gpu==2.0.0b1"
$ pip install -q -U tensorflow_hub
Images for Testing
In the Command Line Interface workspace we have we have provided 4 images in the ./test_images/ folder for you to check your prediction.py module. The 4 images are:

cautleya_spicata.jpg
hard-leaved_pocket_orchid.jpg
orange_dahlia.jpg
wild_pansy.jpg
