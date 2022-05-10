
# LabelNoise
## Data cleansing

One approach to handle noise label is data cleansing method. We can import and create instance of the LabelNoise class and apply the data cleansing method on the noisy data to remove and reassign new suggested labels. Data cleansing method will identify label noise by running classification mutliple time iteratibly on the whole dataset which contains noise in cross-validated form and reasign new labels to noisy data using a model that trained with cleansed data. The method continues till the `noise_ratio` of the dataset drops below some threshold (default is 0.05). The method described in more detail in our paper <a href="https://doi.org/10.1016/j.bpsc.2020.05.008">[1]</a>.

Caveate: the choise of classifier and its parameters is important for datacleansing. The classifier and in general the pipeline should be able to classify with good accuracy. One way to obtain good candidate for classification pipeline is to apply grid search on multiple models with different range of parameters on original data. Having a good classifier that can classify the noise free dataset is the main assumption of this method. 

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
