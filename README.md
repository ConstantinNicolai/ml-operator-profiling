# ml-operator-profiling

To replicate please use the provided conda environment

The idea is to use either nvidia-smi or nvitop to continously monitor energy and time for a yet to be determined set of pytorch machine learning operators and measure their cost in terms of time and energy. These measurements will be collected in a databse to be used shortly after in my master thesis work. 

### operator configrurations
For a pytorch operator such as conv2d a number or configurations can be tested. There can be different sizes, sparsities, kernel sizes, step sizes etc.

### measurement methodology approach
For each operator/ configuration of said operator a measurement run is to be performed in which I intend to call the operator continously in a loop for an extended period of time. To create a reasonable caching scenario I will create a large array of random data of a fitting data type and move that to gpu memory. Then I will call the operator to work on vectors from that array in a linearly indexed fashion. In the case that the array is not large enough to allow for my desired runtime of the experiment I will loop over the array.

### averaging of results
To find final values for the measurent I will do one preliminary experiment where I calculate the median and I calculate the arithmetic mean, disregarding all values outside a 3 sigma sphere. If these values correspond sufficiently well I will decide on one of them as the methodology for the study.

### time resolution requirements - longer runtime measurement ?
Is it a sensible approach to compensate fot the limited time resolution of out energy measurements by just running a large N of repitions with a lower, known good time resolution and averging these measurements afterwards ?