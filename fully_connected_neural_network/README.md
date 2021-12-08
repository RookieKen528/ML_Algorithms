Fully Connected Neural Network (FNN)

this is a simple implementation of FNN for learning purpose.

- using Stochastic Gradient Descent (SGD) and backprop algorithm.
- hidden layer activation currently support only ReLu.
- output layer activation currently support only sigmoid.
      

Interface:

- add_layer(), add_output_layer() method can be used to construction the architecture of the neural network.
- Input() method can be used to input training data.
- train() method is used to start the training process.
- predict() method is used to calculate prediction.
- export_model() method can be used to export parameters of a trained network for later model reconstruction.
- load_model() method can be used to reconstruct of model.

User can also play around with other internal use method, such as:

- forward_prop() : forward pass algorithm
- backward_prop() : backward propagation algorithm        



Email: ken1997528@hotmail.com

WeChat: 799557629

