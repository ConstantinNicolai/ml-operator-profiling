# ml-operator-profiling

## Today's To do's
Read this paper: Inspired by: Lorenz Braun, Sotirios Nikas, Chen Song, Vincent Heuveline and Holger Fröning, A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels - see website \
Measure the dataset for conv2D with a sensible, but moderate amount of configurable parameters. Then perform an energy comparison with ResNet50. Done. As expected by only measuring the conv2D layers we measure less energy in the addition of those then for the full model. Next we should also measure some linear layers we need for the resnets. \
We also should add a pipeline to read in the model analysis and add up the layers in a model according to the number of their incidences. If we can build this in a sufficiently generalized manner this should allow us to study many different models. \
Where are we? Initial pipelines for resnets and conv2d work. This is a promising prove of concept. Now we need to generalize. \
Lets not do the approach with print(model) and torchsummary and instead work with the pytorch objects themsleves. Meaning we want to store the layer objects themselves and also profile them with the the register_module_forward_hook Kevin suggested. This way we should probably be able to get both the pytorch layer object as well as the inputs and outputs. \
Build a foward hook which checks whether the layer it is at has children = None wich would indicate it being a leaf of the model tree and therefore a layer we actually want to measure. \
We now have a forward hook and code to move all layers into a dictionary which also counts their number of occurences in a given model. Leaning into the infrastructure as code direction, I have decided to have the measurement configs in so called summary files in there measurement folders. These are yml files containing: model, weights, input size and a marker whether they were already processed. The next consideration is whether to make the selection of which operators are significant on a per model or a global basis. Since this is both easier to implement right now and better reflects the global nature of the final database I will go with the global approach.\
A lot of updates need to be written in here, but before I forget, in the current version, the input tensor is not iterated only the operators. 


To replicate please use the provided conda environment "constabass.yml"

The idea is to use either nvidia-smi or nvitop to continously monitor energy and time for a yet to be determined set of pytorch machine learning operators and measure their cost in terms of time and energy. These measurements will be collected in a databse to be used shortly after in my master thesis work. 


### workflow considerations
For any pytorch model we want to study we need to do an initial human assessment defining which layer types we expect to be sigificant contributors in terms of runtime and energy consumption. We also need to set an accuracy target, e.g. we can attribute 90% of the full model runtime and energy to measured operators. We then measure and compare. If we do not reach our accuracy target, we add more operators to the significant contributors list. We measure again. This process is repeated until we reach the desired accuracy. This way we should be able to measure all significant operators which are used in the pytorch models we study for our dataset.

### operator configrurations
For a pytorch operator such as conv2d a number or configurations can be tested. There can be different sizes, sparsities, kernel sizes, step sizes etc.

### measurement methodology approach
For each operator/ configuration of said operator a measurement run is to be performed in which I intend to call the operator continously in a loop for an extended period of time. To create a reasonable caching scenario I will create a large array of random data of a fitting data type and move that to gpu memory. Then I will call the operator to work on vectors from that array in a linearly indexed fashion. In the case that the array is not large enough to allow for my desired runtime of the experiment I will loop over the array.

### the shared memory conundrumm
I will have to do experimantation on that to be certain, but I think there is an inherent flaw in the methodologgy approach I layed out above. You see, the idea is to measure operators power and time usage for individual pytorch ML operators. If done sufficiently rigorously this should allow me to add up all the power and time costs from the operations in a ML model and have them add up to roughly the meadured values for the continous execution of said model.
Now with my loop approach and using random vectors from gpu global mmemory for the input feature map of each convolution I load this ifmap from global memory every time. In case of the execcution of a ResNet for example I would hypothisise that the ofmap of a layer being the ifmap of the next layer is held in gpu shared memory leading to lower access time and less power usage through data movement. Perhaps it is also just a part of the ifmap being held there. In order to combat this issue I propose one of these two approaches. Either I still use our random convolutional layers but use the same ifmap over and over again, paying this will be kept in shared memory by pytorch. Or I build "netowrks" from identical paramter random conv layers that conserve ifmap ofmap dimensions and measure in that way. This does have the disadvantage of not allwoing for all paramter commbination though since it neccesitates conservation if dimensionality..\
Resusing the ifmap has changed the gpu utilization from 40% max to 100%. This leads me to believe, that this is a more representative approach to measuring the operators.

### averaging of results
To find final values for the measurent I will do one preliminary experiment where I calculate the median and I calculate the arithmetic mean, disregarding all values outside a 3 sigma sphere. If these values correspond sufficiently well I will decide on one of them as the methodology for the study.

### time resolution requirements - longer runtime measurement ?
Is it a sensible approach to compensate fot the limited time resolution of out energy measurements by just running a large N of repitions with a lower, known good time resolution and averging these measurements afterwards ?
We dont't really care about the time resolution, but it is desirable to have relatively short measurements, to allow for more measurements with different operator configurations and different operators.
