# Experiments with Facebook SwAV (Swapping Assignments between multiple Views of the same image)


This repo is a set of experiments using SwAV (Swapping Assignments between multiple Views of the same image). SwAV is an online clustering method for unsupervised representation learning in images. The research is published by the Facebook AI Research group and Inria centre at the University Grenoble Alpes. Accompanying the paper is the code distributed under the Creative Commons license.

```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
This workspace is a clone of https://github.com/facebookresearch/swav.git with changes and experiments. Because the facebook code is licensed under the Creative Commons the code here is thus "adapted material" under that license.

In the top-level folder are folder with experiment results, a conda version file, and VS-Code launch.json file. In the swav/ folder is the source code, primarily re-used from Facebook with modifications to suit my simulation platform.

the folders are /Annodataions, /Data, /ImageSets.

/Annotations/train, /Data/train have a senset-based folder structure - the image/annotation is under a folder with a senset prefix. This makes them directly readable by pytorch 
/Data has a "senset" folder structure includes /train, /val, /test. E

Chris Winsor

