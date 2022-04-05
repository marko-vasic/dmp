# Deep Molecular Programming

## Overview
Project allows training chemical networks using deep learning. It
trains a neural network and translates it to an equivalent chemical
network. This is done based on the tight connection, between chemical
and neural models of computation, that we discovered.

Two publications are result of this project: 

```
@article{pnas22TrainingCRNs,
  title = {Programming and Training Rate-Independent Chemical Reaction Networks},
  author = {Vasic, Marko and Chalk, Cameron and Luchsinger, Austin and Khurshid, Sarfraz and Soloveichik, David},
  journal = {Proceedings of the National Academy of Sciences},
  year = {2022}
}

@inproceedings{icml20DeepMolecularProgramming,
  title={{D}eep {M}olecular {P}rogramming: {A} {N}atural {I}mplementation of {B}inary-{W}eight {R}e{L}{U} {N}eural {N}etworks},
  author = {Vasic, Marko and Chalk, Cameron and Khurshid, Sarfraz and Soloveichik, David},
  booktitle = {International Conference on Machine Learning},
  year = {2020},
}
```
If you would like to reference them in an academic publication please
cite the previous papers.

Our **YouTube Presentations** of associated papers:
- [Programming and Training Rate-Independent Chemical Reaction Networks](https://www.youtube.com/watch?v=OWtrPTaIvXM)
- [Deep Molecular Programming](https://www.youtube.com/watch?v=kf-0FLZyoNk)

## Requirements for executing code
- Following software is needed to run the code:
    * Theano (0.7.0 or higher)
    * Lasagne (0.1 or higher)
    * pylearn2 (0.1.dev0)
    * Mathematica (11.2 or higher)

## Running Code
- Note that neural network models (used in the publications mentioned
  above) as well as translated chemical networks are included in the
  repo and are ready to be used. Pretrained neural networks are saved
  under
  [data-repo/models](https://github.com/marko-vasic/dmp/tree/main/data-repo/models)
  directory while chemical networks obtained by translating those are
  saved under
  [data-repo/mathematica](https://github.com/marko-vasic/dmp/tree/main/data-repo/mathematica)
  directory. Thus, one can skip steps 1 and 2 below and go directly to
  step 3 of simulating existing chemical networks.

- Navigate to the *src* directory.

- **Step 1: Training.** To train a neural network run: ```python -m
  crn.subject --train```; where currently supported subjects are:
  *iris*, *virus*, *pattern_formation**, *mnist-subset*, *mnist*.

- Model will be saved under
  [data-repo/models](https://github.com/marko-vasic/dmp/tree/main/data-repo/models)
  directory with *pkl* extension.

- **Step 2: Translation.** After training a model you can translate it
  to a CRN by running ```python -m crn.subject --translate```.

- Translated CRN will be stored under
  [data-repo/mathematica](https://github.com/marko-vasic/dmp/tree/main/data-repo/mathematica)
  directory with *wls* extension (wls is a Mathematica script file).

- **Step 3: Chemical Simulations.** Finally, you can run Mathematica
  simulations by navigating to
  [data-repo/mathematica](https://github.com/marko-vasic/dmp/tree/main/data-repo/mathematica),
  and executing the produced wls file.

- Kinetics simulations of the produced CRN will be stored under
  [data-repo/kinetics](https://github.com/marko-vasic/dmp/tree/main/data-repo/kinetics)
  directory.

- *Possible issues:* Note that MNIST neural network model (mnist.pkl)
  might fail to translate on some systems due to compatibility
  issues. This shouldn't prevent you to use CRN obtained from
  translating that neural network, which we saved in mnist.wls
  file. We are working on translating model files to a new, more
  portable format.

## Acknowledgments
Some of the graphics in pattern formation dataset
([data-repo/datasets/pattern_formation/graphics](https://github.com/marko-vasic/dmp/tree/main/data-repo/datasets/pattern_formation/graphics))
are created by [Joseph Wain](http://penandthink.com) and licensed
under [CC BY 3.0 US](http://creativecommons.org/licenses/by/3.0/us/).

To train binary neural networks we adapt
[BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect)
code. We augment BinaryConnect to support zero weights. BinaryConnect
code resides in
[src/binary_connect](https://github.com/marko-vasic/dmp/tree/main/src/binary_connect)
directory.
