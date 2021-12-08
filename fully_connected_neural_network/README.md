# Fully Connected Neural Network (FNN)

## Description

This is a simple implementation of FNN for **learning purpose**.

- using Stochastic Gradient Descent (SGD) and backprop algorithm.
- hidden layer activation currently supports only ReLu.
- output layer activation currently supports only sigmoid.

## File:

- FNN.ipynb : jupyter notebook version with explanation of the code and the math
- fnn.py : python file that only contains the FNN class
- data_banknote_authentication.txt : toy data used in FNN.ipynb. The data set consists of 1372 examples, each example representing a banknote. There are four features. The goal is to predict if a banknote is authentic (class 0) or a forgery (class 1).
      

## Interface:

- add_layer(), add_output_layer() method can be used to construction the architecture of the neural network.
- Input() method can be used to input training data.
- train() method is used to start the training process.
- predict() method is used to calculate prediction.
- export_model() method can be used to export parameters of a trained network for later model reconstruction.
- load_model() method can be used to reconstruct of model.

User can also play around with other internal use method, such as:

- forward_prop() : forward pass algorithm
- backward_prop() : backward propagation algorithm        

## Contact

Email: ken1997528@hotmail.com

WeChat: 799557629

