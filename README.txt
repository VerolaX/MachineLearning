Tianyou Xiao
2985532  txiao3
CSC246 HW2

This homework requires us to implement the perceptron algorithm based on NumPy.

Submitted files:
Xiao_perceptron.py -- source code for this homework
test.png -- graph showing some tests based on different numbers of iterations and learning rates
README.txt

My model passes the smoke test and two test cases provided on the website. I also did some experiments on different numbers of iterations (up to 100) and different learning rates ([0.001, 0.01, 0.05, 0.1, 1]). 
The test plot shows how the models perform on training set and development set change as number of iterations changes/ learning rates.
From the graph, we can tell that the model converges extremely fast (almost converges in the first iteration), and the accuracies just fluctuate within a small range as the number of iterations increases.
Meanwhile, the value of learning rates also does not affect the accuracies much. Accuracies of all combinations of iterations and learning rates remain around 0.75-0.83.
There is also almost no difference between the accuracies between the performances on training data and development data. I guess the reason is that the dataset is just too small.

If "--nodev" argument is provided, development data will not be used. Otherwise, the algorithm will evaluate how the model performs after each iteration and save the best model with its iteration number. The function will ultimately return the best model and indicate the best iteraton number as well.

