# ml-operator-profiling

To replicate please use the provided conda environment

The idea is to use either nvidia-smi or nvitop to continously monitor energy and time for a yet to be determined set of pytorch machine learning operators and measure their cost in terms of time and energy. These measurements will be collected in a databse to be used shortly after in my master thesis work. 

### operator configrurations
For a pytorch operator such as conv2d a number or configurations can be tested. There can be different sizes, sparsities, kernel sizes, step sizes etc.

### measurement methodology approach
For each operator/ configuration of said operator a measurement run is to be performed in which I intend to call the operator continously in a loop for an extended period of time. The startup and shutdown periods will be disregarded due to, yes, due what actually ?
