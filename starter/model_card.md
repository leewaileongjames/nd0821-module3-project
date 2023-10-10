# Model Card

## Model Details

The model is a K-Nearest-Neighbors classifier from sklearn.neighbors.KNeighborsClassifier.

The 'n_neighbors' parameter was set at '5', with the remaining parameters at default.

<br/>

## Intended Use

The model is meant to predict if the annual salary of an individual might be more or less than $50k.

<br/>

## Training Data

80% of the cleaned census data (sampled at random) was used to train the model.

| Dataset Characteristics  | Attribute Types  | No. of Entries | No. of Attributes | Missing Values | 
|---|---|---|---|---|
| Multivariate | Categorical, Numeric | 32561 | 14 | Yes |

During data cleaning,
- Entries containing missing values were dropped. As a result, only 30162 entries were subsequently used for training and testing.
- The 'fnlgt' attribute was also dropped as there was insufficient information available on this attribute

More information on the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

<br/>

## Evaluation Data

The remaining 20% of the cleaned census data was used to evaluate the performance of the trained model.

<br/>

## Metrics

**Precision**: Precision is a measure of the accuracy of a classifier when it predicts a positive result.

- Precision = TP / (FP + TP)

**Recall**: Recall, also known as sensitivity or true positive rate, measures the ability of a classifier to identify all positive instances.

- Recall = TP / (FN + TP)

**F-Beta score**: F-beta score is a measure that combines both precision and recall into a single number.

- F-Beta score = (1+β^2)(precision*recall) / ((β^2)precision+recall)

<br/>

Performance of the model evaluated against the evaluation data:

| Metric  | Value  |
|---|---|
| Precision | 0.7207142857142858 |
| Recall    | 0.6594771241830065 |
| F-Beta    | 0.6887372013651877 | 

<br/>

## Ethical Considerations

During model training, there was no investigation done in terms of bias and there was no hyperparameter tuning being performed. Therefore, predictions obtained from the trained model is not guaranteed to be fair.

<br/>

## Caveats and Recommendations

The data was extracted from the 1994 Census database. Therefore, there is a high chance that the model will not perform as well when applying it on the real world today. 

Furtheremore, the metrics gathered on various data slices suggests that model performance is not uniform across different categories of the population.

