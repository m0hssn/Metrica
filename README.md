# Metrica
The `Metrica` class provides a way to store predicted and ground truth values for a classification task and calculate various metrics such as F1 score, accuracy, recall score, precision score, AUROC and AUPRC based on these values. It uses the `torchmetrics` library to calculate these metrics.

## Usage
To use the `Metrica` class, first create an instance by specifying the number of classes for the classification task and the type of classification task (either “multiclass” or “binary”):

```python
from metrica import Metrica

num_classes = 3
task = "multiclass"

metrica = Metrica(num_classes=num_classes, task=task)
```

Once you have created an instance of the Metrica class, you can use the `upgrade` method to add new predicted and ground truth values:

```python
y_hat = ...  # predicted values
y = ...  # ground truth values

metrica.upgrade(y_hat, y)
```

You can then use the various methods of the `Metrica` class to calculate different metrics based on the stored predicted and ground truth values:

```python
f1_score = metrica.f1_score()
accuracy = metrica.accuracy()
recall_score = metrica.recall_score()
precision_score = metrica.precision_score()
auroc = metrica.AUROC()
auprc = metrica.AUPRC()
```

You can also use the `reset` method to clear the stored predicted and ground truth values:

``` python
metrica.reset()
```

And finally, you can use the `print_metrics` method to print all the calculated metrics in one line:

```python
metrica.print_metrics()
```

Example
Here’s an example that shows how to use the `Metrica` class to store predicted and ground truth values for a classification task and calculate various metrics based on these values:

```python
from metrica import Metrica

# Create an instance of the Metrica class
num_classes = 3
task = "multiclass"
metrica = Metrica(num_classes=num_classes, task=task)

# Add some predicted and ground truth values
y_hat = ...  # predicted values
y = ...  # ground truth values
metrica.upgrade(y_hat, y)

# Calculate some metrics
f1_score = metrica.f1_score()
accuracy = metrica.accuracy()
recall_score = metrica.recall_score()
precision_score = metrica.precision_score()
auroc = metrica.AUROC()
auprc = metrica.AUPRC()

# Print the calculated metrics
metrica.print_metrics()

# Reset the stored predicted and ground truth values
metrica.reset()
```

This example creates an instance of the `Metrica` class, adds some predicted and ground truth values using the `upgrade` method, calculates various metrics using the corresponding methods of the `Metrica` class, prints the calculated metrics using the `print_metrics` method, and finally resets the stored predicted and ground truth values using the `reset` method.
