Tianyou Xiao
2985532  txiao3
CSC246 HW3

This homework requires us to implement forward and backpropagation of a neural network based on NumPy.

Submitted files:
Xiao_backprop.py -- source code for this homework
hid_2.png, hid_5.png, hid_10.png, hid_20.png -- graphs showing accuracies with different combinations of learning rates, iteration numbers and hidden layer dimensions. hid_X.png shows the accuracies when there are X neurons in the hidden layer.
README.txt

My model passes the smoke test. I also did some experiments on different numbers of iterations (up to 100), different learning rates ([0.001, 0.05, 0.01, 0.05, 0.1, 1]) and different sizes of hidden layers ([2,5,10,20]). The hid_X plots show how the models perform on training set and development set as iteration number/learning rates change, based on different hidden layer size.

From the graphs, we can see that the size of hidden layer matters. In general, smaller hidden layer dimension indicates faster convergence of the model. For example, when hidden_dim=2, models with different learning rates perform reasonably well at the beginning of the training process, and may improve a little bit after more iterations. Meanwhile, when hidden_dim=20, models with small weights take longer time to actually obtain a good predicting accuracy. Different learning rates also affect the models a lot. When the network's structure becomes complicated, models with smaller learning rates may take longer time than those with larger learning rates to converge. For the number of iterations, there is no guarantee that the model will converge at one particular iteration all the time, since the weights are randomly initialized, and it also depends on the learning rates and hidden layer size.

Compared with the perceptron algorithm we implement for HW2, the overall performance of neural network, which usually produces about 85% of predicting accuracy, is slightly better in terms of predicting accuracy, but not by too much.
 
If "--nodev" argument is provided, development data will not be used. Otherwise, the algorithm will evaluate how the model performs after each iteration and save the best model with its iteration number. The function will ultimately return the best model and indicate the best iteration number as well.

