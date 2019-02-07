Tianyou Xiao
2985532  txiao3
CSC246 HW2

This homework requires us to implement forward and backpropagation of neural network based on NumPy.

Submitted files:
Xiao_backprop.py -- source code for this homework
hid_2.png, hid_5.png, hid_10.png, hid_20.png -- graphs showing accuracies with different combinations of learning rates, iteration numbers and hidden layer dimensions. hid_X.png shows the accuracies when there are X neurons in the hidden layer.
README.txt

My model passes the smoke test. I also did some experiments on different numbers of iterations (up to 100), different learning rates ([0.001, 0.05, 0.01, 0.05, 0.1, 1]) and different sizes of hidden layers ([2,5,10,20]). The hid_X plots show how the models perform on training set and development set as iteration number/learning rates change, based on different hidden layer size.

From the graphs, we can see that the size of hidden layer does matter. Generally, smaller hidden layer dimension indicates faster convergence of the model. For example, when the hidden_dim=2, models with different learning rates perform well at the beginning of the training process, and improve a little bit after more iterations.

Compared with the perceptron algorithm we implement for HW1, the overall performance of neural network is slightly better in terms of prediction accuracy, but not by too much.
 
If "--nodev" argument is provided, development data will not be used. Otherwise, the algorithm will evaluate how the model performs after each iteration and save the best model with its iteration number. The function will ultimately return the best model and indicate the best iteration number as well.

