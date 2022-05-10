
# LabelNoise
## Data cleansing

One approach to handle noise label is data cleansing method. We can import and create instance of the LabelNoise class and apply the data cleansing method on the noisy data to remove and reassign label noise. Data cleansing method will identify label noise base on running classification mutliple time iteratibly on the noisy data in cross-validated form and reasign new labels for noisy data using a model trained via cleansed data. The method continues till the noise_ratio of the dataset drops below some threshold (default is 0.05). The method described in more detail in our paper [1].

Caveate: the choise of classifier and its parameters is important. One way to obtain good candidate for classification pipeline is to apply grid search on multiple models with different range of parameters on original data.

The LabelNoise class can take:
```
data: input data
targe: the given labels
name: name for instance.
pipeline: classification pipeline
cv: cross validation form
balance_method: the method to handel imbalanced classes. undersampling, oversampling and None.
noise_threshold: onise ratio threshold to stop convergance of data cleansing algorithm.
```

After creating instance we can call `datacleansing` function of the instance to identify and clean label noises. The function will returns the `given_labels` and `cleansed_labels` for the dataset.

## Demo
<a href="https://github.com/hoomanro/LabelNoise/blob/main/datacleansing_example.ipynb">example of this datacleansing method on handwriting digits dataset</a>

## References
[1] Rokham, Hooman, et al. "Addressing inaccurate nosology in mental health: A multilabel data cleansing approach for detecting label noise from structural magnetic resonance imaging data in mood and psychosis disorders." Biological Psychiatry: Cognitive Neuroscience and Neuroimaging 5.8 (2020): 819-832. https://doi.org/10.1016/j.bpsc.2020.05.008
