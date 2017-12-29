# Image-Classification-using-Machine-Learning
Implementation of Neural Networks, Gradient boosting and KNN for predicting correct image orientation.

USAGE:  
Training: 
./orient.py test test_file.txt model_file.txt [model]
where [model] is again one of nearest, adaboost, nnet, best.

Testing: 
./orient.py test test_file.txt model_file.txt [model]
where [model] is again one of nearest, adaboost, nnet, best.

• nearest: At test time, for each image to be classified, the program should find the k nearest images in the training file, i.e. the ones with the closest distance (least vector diference) in Euclidean space,and have them vote on the correct orientation.

• adaboost: Use very simple decision stumps that simply compare one entry in the image matrix to another, e.g. compare the red pixel at position 1,1 to the green pixel value at position 3,8. You can try all possible combinations (roughly 1922) or randomly generate some pairs to try.

• nnet: Implement a fully-connected feed-forward network to classify image orientation, and implement the backpropagation algorithm to train the network using gradient descent.

• best:Recommended algorithm that gives the best accuracy.

This program loads in the trained parameters from model file.txt, run each test example through the model, display the classification accuracy (in terms of percentage of correctly-classified images), and output a file called output.txt which indicates the estimated label for each image in the test file. The output file should cooresponds to one test image per line, with the photo id, a space, and then the estimated label, e.g.:
test/124567.jpg 180
test/8234732.jpg 0

We have two ASCII text files, one for the training dataset and one for testing, that contain the feature vectors. 
To generate this file, we rescaled each image to a very tiny micro-thumbnail of 8 * 8 pixels, resulting in an 8 * 8 * 3 = 192-dimensional feature vector.

The text files have one row per image, where each row is formatted like:  
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...  
where:  
• photo id is a photo ID for the image.  
• correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be wrong because of noise; this is just a fact of life when dealing with data from real-world sources.  
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc.each in the range 0-255.
 
You can view the original high-resolution image on Flickr.com by taking just the numeric portion of the photo id in the file above (e.g. if the photo id in the _le is test/123456.jpg, just use 123456), and then visiting the following URL: 
http://www.flickr.com/photo_zoom.gne?id=numeric photo id


Please refer the report for further implementation details.
