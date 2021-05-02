<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

This repo implements AdaSwarm, an optimizer, that combines Gradiant Descent and Particile Swarms. 

AdaSwarm is based on "AdaSwarm: Augmenting Gradient-Based optimizers in Deep Learning with Swarm Intelligence, _Rohan Mohapatra, Snehanshu Saha, Carlos A. Coello Coello, Anwesh Bhattacharya Soma S. Dhavala, and Sriparna Saha_", to appear in IEEE Transactions on Computational Intelligence.

An arXive version is [here](https://arxiv.org/abs/2006.09875)


[This](https://github.com/rohanmohapatra/pytorch-cifar) repo containes implemenation used the paper.

[[1]](#1)

## Why AdaSwarm:
Said  et  al.  [[1]](#1)  postulated  that  swarms behaviour is similar to  that of classical  and  quantum  particles.  In  fact, their analogy is so striking that one may think that the social and  individual  intelligence  components  in  PSO  are,  after  all, nice useful metaphors, and that there is a neat underlying dynamical system at play. This dynamical system perspective was indeed useful in unifying two almost parallel streams, namely, optimization  and  Markov  Chain  Monte  Carlo  sampling. In a seminal paper, Wellington and Teh [[2]](#2), showed that a  stochastic  gradient  descent  (SGD)  optimization  techniquecan  be  turned  into  a  sampling  technique  by  just  adding noise,  governed  by  Langevin  dynamics.  Recently,  Soma  andSato [[3]](#3) provided further insights into this connection based on  an  underlying  dynamical  system  governed  by  stochastic differential equations (SDEs). While these results are new, the connections  between  derivative-free  optimization  techniquesbased on Stochastic Approximation and Finite Differences are well documented [[4]](#4). Such strong connections between these seemingly  different  subfields  of  optimization  and  sampling made  us  wonder:  Is  there  a  larger,  more  general  templateof  which  the  aforementioned  approaches  are  special  cases. AdawSwarm is a result of that deliberation. We believe that, it is just a beginning of a new breed of **composable optimizers**

## What is AdaSwarm in simple terms, in the context Deep Learning:
1. Setup
    - "y": responses
    - "f(.)" is a model specified by a network with parameters "w"
    - "f(x)" is the prediction at observed feature x
    - "L(.)" is loss, to drive the optimization

2. Approximate gradiants of "L(.)" w.r.t "f(.)"
    - run an independent Sawrm optimizer over L with particlae dimension equal to the size of the network's output layer
    - using parotcile parameters, approximate the gradiant of  "L(.)" w.r.t "f(.)"

3. Get gradiants of "f(.)" w.r.t "w"
    - using standard "AutoDiff", via chain rule, get the approximate gradiant of "f(.)" w.r.t "w"

4. Approximate gradiants of "L(.)" w.r.t "w"
    - take the product of gradiants in steps (2) and (3)

5. Updates the network weigths via standard Back Propagation

## Why does it work? Changes are very negligible

At this time, we only have huch that, Swarm, by being a meta-hueristic algorithm, has less tendency to trap in local minimal or has better exploration capabilities, is helping the problem overall. Secondly, entire information about the "learning" comes from the loss, and the function "f(.)" only specifies the relationship between input and output. So, having better "optimization" capabilities at the loss, in general, are going to be helpfu. We have ample empirical evidence to support this claim. We dont have rich theory, however. In contrast, to the best of our knowledge, all current optmization techniues, only harvest information coming from a single paradigm. AdaSwarm combines different perspectives, like in an ensemble. More of an ensemble, it is a composition -- where different perspectives get added. In someways, just like an neural network archicture is compsoed of several layers, AdaSwarm is a composition of optmizers. That composition eventually fits into the chain rule.

The changes are very small. Same is the case with Adam and RMSProp, right? Another examples are that of the prnounced differences in speed/convergance, with very simple changes in the maps: case in point are
- "Proximal gradient descent" vs "Accelerated Proximal gradiant descent"
- "Adam" vs "RMSProp"
- "Euler" vs "LeapFrog" ...
We have to develop both theoreical and computational tools to understand why and where AdaSwarm works.

## Objectives:

1. Develop a plug-and-play optimizer that works with
    - other optimizers in the PyTorch ecosystem, along side the likes of "Adam", "RMSProp", "SGD"
    - any archictecure 
    - any dataset
    - with the same api as others, i.e., "optim.AdaSwarm()"

2. Battle test on variety of
    - test objectives functions
    - datasets
    - architectures (Transformers, MLPs,..)
    - losses (BCE, CCE, MSE, MAE, CheckLoss,...)
    - paradigms (RIL, Active Learning, Supervised Learning etc..)
    - etc..

3. Provide insighhts into the workings of AdaSwarm by
    - analyzing the workings of the optimizers
    - Visualizating the path trajectories
    - etc..

4. Most importantly, be community driven
    - there is lot of enthusism and interest in the recent graduates and undegraduates, that want to learn ML/AI technologies. Instead of fiddling with MNSIT datasets, and predicting Cats and Dogs, do something foundational and meaninful.
    - turn this into a truely community-driven effort to offer a useable, useful, foundational building block to the deep learning ecosystem



## Contributing:
While we are yet to establish the policty to contribute, we will follow how any Apcahe open source project wotks. For example, see airflow project's [contribution](https://github.com/apache/airflow/blob/master/CONTRIBUTING.rst) guidelines. 


## References
<a id="1">[1]</a> 
S. M. Mikki and A. A. Kishk,Particle Swarm Optimizaton: A Physics-Based Approach.    Morgan & Claypool, 2008.
<a id="2">[2]</a> 
M.  Welling  and  Y.  W.  Teh,  “Bayesian  learning  via  stochastic  gradientlangevin dynamics,”In Proceedings of the 28th International Conferenceon Machine Learning, p. 681–688, 2011.
<a id="3">[3]</a> 
Y.-A. Ma, Y. Chen, C. Jin, N. Flammarion, and M. I. Jordan, “Samplingcan be faster than optimization,”Proceedings of the National Academyof   Sciences,   vol.   116,   no.   42,   pp.   20 881–20 885,   2019.   [Online].Available: https://www.pnas.org/content/116/42/20881[21]  S.  Yokoi  and  I.  Sato,  “Bayesian  interpretation  of  sgd  as  ito  process,”ArXiv, vol. abs/1911.09011, 201.
<a id="3">[4]</a> 
J.  Spall, Introduction  to  stochastic  search  and  optimization.Wiley-Interscience, 2003

## Citation

AdaSwarm will be appearing in the [paper](https://arxiv.org/abs/2006.09875) you can cite:
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