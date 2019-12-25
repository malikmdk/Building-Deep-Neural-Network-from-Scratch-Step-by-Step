# Building-Deep-Neural-Network-from-Scratch-Step-by-Step
I implement deep neural network from scratch using numpy. 

## Introduction
To build deep neural network, we will be implementing several "helper functions". These helper functions will be used to build an L-layer deep neural network.

- Initialize the parameters for  an $L$-layer deep neural network.
- Implement the forward propagation module (shown in purple in the figure below).
     - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
     - We give you the ACTIVATION function (relu/sigmoid).
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in the figure below).
    - Complete the LINEAR part of a layer's backward propagation step.
    - We have to implement gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Update the parameters.
- Finally combine all these helper function in model function.


**For futher detail please visit notebook available in repositry. All these funtion are implemented with well commited code.**

## Reference
> coursera deep neural network specilization.
