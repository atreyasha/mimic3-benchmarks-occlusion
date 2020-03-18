Developments
------------

### Feature importance via **occlusion**

1.  **TODO** 4 occlusion types on right, wrong, 2 classes
    (possibly permutated with right/wrong) and overall analysis

2.  **TODO** occlude by permutating all features indepedently
    across whole dataset, or occlude by class on purpose

3.  visualize results using full test dataset, or break down analysis
    into correct/wrong predictions and also possibly into class-wise
    importance

### Analysis on new model

1.  **TODO** focus on lstm model, either from Oguz or from
    mimic3 depending on what is an easier start

2.  consider practical packages for helping with this -\> eg. lime,
    eli5, alibi, lucid (mainly for NN), DeepExplain, innvestigate;
    additionally book by Christoph and digital signal processing summary
    of explainability methods

### Presentation and code-health

1.  port to pytorch for simplicity and more widespread porting

### Feature importance via **gradient** methods and additional information

1.  look at gradient based approaches recommended by David

2.  read through book and write out pro\'s and con\'s of various
    available feature importance techniques -\> summarize and present 3
    effective and practical methods we could try; with mentioning
    feature masking/permutation for better context

3.  model variance and feature importance correlate for general models
    -\> so might be better to explore both model variance and
    performance drop simultaneously to understand the background instead
    of assuming

4.  choose train/test data depending on the end task -\> introspect
    train model or introspect how it handles unseen data

5.  read more on exact procedure of occlusion and best practices; such
    as occluding train or test data, and how to occlude ie. mask vs.
    randomize

6.  be careful with variance of features as they should be consistent

### Higher-level ideas

1.  use existing deep NN explainability methods to approximate global
    feature importance for mimic3 benchmarks and possibly our new paper,
    to justify which advantages our network has and what it considers
    correctly

2.  makes sense to use more than just occlusion, to get diverse input
    features to check for consistency
