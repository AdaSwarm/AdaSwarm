<!---
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# AdaSwarm

This repo purportedly implements *AdaSwarm*, an optimizer, that combines Gradient Descent and Particle Swarms. 

*AdaSwarm* is based on "AdaSwarm: Augmenting Gradient-Based optimizers in Deep Learning with Swarm Intelligence, _Rohan Mohapatra, Snehanshu Saha, Carlos A. Coello Coello, Anwesh Bhattacharya Soma S. Dhavala, and Sriparna Saha_", to appear in IEEE Transactions on Computational Intelligence. An arXive version can be found [here](https://arxiv.org/abs/2006.09875). [This](https://github.com/rohanmohapatra/pytorch-cifar) repo contains implementation used the paper.


## Why *AdaSwarm*:
Said  et  al.  [[1]](#1)  postulated  that  swarms behavior is similar to  that of classical  and  quantum  particles.  In  fact, their analogy is so striking that one may think that the social and  individual  intelligence  components  in  Swarms  are,  after  all, nice useful metaphors, and that there is a neat underlying dynamical system at play. This dynamical system perspective was indeed useful in unifying two almost parallel streams, namely, optimization  and  Markov  Chain  Monte  Carlo  sampling. 

In a seminal paper, Wellington and Teh [[2]](#2), showed that a  stochastic  gradient  descent  (SGD)  optimization  technique can  be  turned  into  a  sampling  technique  by  just  adding noise,  governed  by  Langevin  dynamics.  Recently,  Soma  and Sato [[3]](#3) provided further insights into this connection based on  an  underlying  dynamical  system  governed  by  stochastic differential equations (SDEs). 

While these results are new, the connections  between  derivative-free  optimization  techniques based on Stochastic Approximation and Finite Differences are well documented [[4]](#4). Such strong connections between these seemingly  different  subfields  of  optimization  and  sampling made  us  wonder:  Is  there  a  larger,  more  general  template, of which  the  aforementioned  approaches  are  special  cases, exist? *AdaSwarm* is a result of that deliberation.

We believe that it is just a beginning of a new breed of **composable optimizers**

## What is *AdaSwarm* in simple terms, in the context Deep Learning:
1. Setup
    - ``y``: responses
    - ``f(.)`` is a model specified by a network with parameters ``w``
    - ``f(x)``is the prediction at observed feature ``x``
    - ``L(.)`` is loss, to drive the optimization

2. Approximate gradients of ``L(.)`` w.r.t ``f(.)``
    - run an independent Swarm optimizer over ``L(.)`` with particle dimension equal to the size of the network's output layer
    - using swarm particle parameters, approximate the gradient of  ``L(.)`` w.r.t ``f(.)``

3. Get gradients of ``f(.)`` w.r.t ``w``
    - using standard ``AutoDiff``, via chain rule, get the approximate gradient of ``f(.)`` w.r.t ``w``

4. Approximate gradients of ``L(.)`` w.r.t ``w`` via Chain Rule
    - take the product of gradients in steps (2) and (3)

5. Updates the network weights via standard Back Propagation

## Why does it work? Changes seem very minor!

At this time, we only have hunch that, Swarm, by being a meta-heuristic algorithm, has less tendency to get trapped in local minimal or has better exploration capabilities. It is helping the problem overall. Secondly, entire information about the "learning" the task comes from the loss, and the function ``f(.)`` only specifies the structural relationship between input and output. 

So, having better "optimization" capabilities at the loss, in general, are going to be helpful. While we have ample empirical evidence that shows that *AdaSwarm* is working well, we don't have rich theory, however. We only have few speculations like the one stated earlier.

Another speculation, speculation at this time, is that, to the best of our knowledge, all current optimization techniques only harvest information coming from a single paradigm. *AdaSwarm*, whereas, combines different perspectives, like in an ensemble. More than an ensemble, it is a composition -- where different perspectives get chained. That is one fundamental difference between *AdaSwarm* and other population-based techniques.

In someways, just like an neural network architecture is composed of several layers, *AdaSwarm* is a composition of optmizers. That composition eventually fits into the chain rule.

As a result, the changes are very small. Same is the case with Adam and RMSProp, right? Other notable examples, where we see pronounced differences in speed/convergence, with very simple changes in the maps are:
- _Proximal gradient descent_ vs _Accelerated Proximal gradient descent_
- _Euler_ vs _LeapFrog_ 
_ ...

Therefore, in order to better understand, and develop the theory and tools for composable optimizers, we have to develop both theoretical and computational tools to understand why and where *AdaSwarm* works. Along the way, make such optimizers accessible to the community.

## Objectives:

1. Develop a plug-and-play optimizer that works with
    - other optimizers in the PyTorch ecosystem, along side the likes of ``Adam``, ``RMSProp``, ``SGD``
    - any architecture 
    - any dataset
    - with the same api as others, i.e., ``optim.AdaSwarm()``

2. Battle test on variety of
    - test objectives functions
    - datasets
    - architectures (Transformers, MLPs,..)
    - losses (BCE, CCE, MSE, MAE, CheckLoss,...)
    - paradigms (RIL, Active Learning, Supervised Learning etc..)
    - etc..

3. Provide insights into the workings of *AdaSwarm* by
    - analyzing the workings of the optimizers
    - visualizing the path trajectories
    - etc..

4. Most importantly, be community driven
    - there is lot of enthusiasm and interest in the recent graduates and undergraduates, that want to learn ML/AI technologies. Instead of fiddling with MNIST datasets, and predicting Cats and Dogs, do something foundational and meaningful. If you take offence to statement, you are not ready for this project.
    - turn this into a truly community-driven effort to offer a useable, useful, foundational building block to the deep learning ecosystem


## Contributing:

1. While we are yet to establish the policy to contribute, we will follow how any Apache open source project works. For example, see airflow project's [contribution](https://github.com/apache/airflow/blob/master/CONTRIBUTING.rst) guidelines. 
 
2. But be mindful. There may not be any short-term rewards. 
    - Research is bloody hard work. There will not be any instant gratification or recognition for the work. Expect lot of negative results, and set backs.
    - Optimization problems are generally hard, and writing an Engineering level framework that works on _any_ problem is even harder. It is scientific computing, not writing hello world examples.
    - So take a plunge only if you are willing to endure the pain, w/o worrying about the end result.


## References
<a id="1">[1]</a> 
S. M. Mikki and A. A. Kishk, Particle Swarm Optimizaton: A Physics-Based Approach.    Morgan & Claypool, 2008.

<a id="2">[2]</a> 
M.  Welling  and  Y.  W.  Teh,  “Bayesian  learning  via  stochastic  gradient langevin dynamics,”In Proceedings of the 28th International Conference on Machine Learning, p. 681–688, 2011.

<a id="3">[3]</a> 
S.  Yokoi  and  I.  Sato,  “Bayesian  interpretation  of  SGD  as  Ito  process,” ArXiv, vol. abs/1911.09011, 201.

<a id="3">[4]</a> 
J.  Spall, Introduction  to  stochastic  search  and  optimization. Wiley-Interscience, 2003

## Citation

*AdaSwarm* will be appearing in the [paper](https://arxiv.org/abs/2006.09875) you can cite:
```bibtex
@inproceedings{adaswarm,
    title = "daSwarm: Augmenting Gradient-Based optimizers in Deep Learning with Swarm Intelligence",
    author = "Rohan Mohapatra, Snehanshu Saha, Carlos A. Coello Coello, Anwesh Bhattacharya Soma S. Dhavala, and Sriparna Saha",
    booktitle = "IEEE Transaction on Computational Intelligence",
    month = tbd,
    year = "2021",
    address = "Online",
    publisher = "IEEE",
    url = "https://arxiv.org/abs/2006.09875",
    pages = tbd
}
```
