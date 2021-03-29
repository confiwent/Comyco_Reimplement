# Comyco_linear_QoE
A re-implemetation of Comyco(ACMMM19' Huang, et al.). The QoE metric is simplified to the linear and objective measurement.

## Requirements
The source codes are based on Python3.6+, with some dependencies: numpy, tensorflow==1.1x, tflearn==0.3.2, sklearn, swig.

## usage
- run ```train_lin.py``` to train the neural network of control policy with imitation learning
- run ```rl_test_lin.py``` to test the model
