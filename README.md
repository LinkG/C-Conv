# C-Conv
Attempting to make a convolutional neural network, without third party libraries. 

The config for the time being must be saved as "network_prop.txt" and must contain the properties "kernel", "num_layers", "layers". All values are comma seperated lists. "kernel" expects a list of 2, "num_layers" expects list of 1, "layers" expects list of size "num_layers". Always have the last layer have a size of 10, to work with numbers.
An example "network_prop.txt" is shown:
```
kernel: 4, 4
layers: 80, 20, 10
num_layers: 3
```
