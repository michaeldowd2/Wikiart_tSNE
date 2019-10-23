# WikiArt Feature Maps

In this repo I've provided a full set of t-SNE embeddings for the wikiart dataset + the code used to create them. Feel free to use them for art or machine learning projects.

The embeddings are built using the keras applications VGG19 convolution network pretrained on the Imagenet dataset to extract two sets of features: one generally representing content, the other style. These features are normalised, weighted and combined before running the sklearn t-SNE over them. 
Here's a diagram of the process architecture:



I was inspired to roll my own t-SNEs by the following projects:
http://smedia.ust.hk/james/projects/deepart_ana.html
https://cs.stanford.edu/people/karpathy/cnnembed/
https://jeremiahwjohnsondotcom.wordpress.com

## Rolling your own t-SNE
Creating them from scratch requires a few steps and preferably a cuda enabled GPU.

## Sample Images and Experiments

