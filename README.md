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

Please refer the report for further details.
