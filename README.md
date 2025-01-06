# Phi3

## AI App using Phi-3  models with OnnxRuntimeGenAI.


## Plant trees:  Get a forest

**Decision trees** can be adjusted to both regression and classification.

Measures for evaluating the performance of decision trees:

1. **Entropy** : Used to measure the impurity or randomness of a dataset.
2. **Gini coefficient** : Probability of a random sample being incorrectly labeled.
3. **Information gain** : Measure of the difference in entropy between the set before and after a split.

All three metrics can be used in decision tree algorithms to determine the best split attribute. 
However, some situations may favor one metric over the others. For example, when dealing with binary classification problems, 
Gini index is preferred over entropy because it tends to be more computationally efficient. 
On the other hand, entropy is preferred when the data set is imbalanced, meaning there is a significant difference in the number of instances belonging to different classes. 
Information gain is a popular metric that is often used because it is easy to understand and generally works well in a variety of situations.

Decision trees contain high variance => with high variance comes the concept of bagging.

**Bagging**
Bagging (bootstrap aggregation) is a procedure used to decrease the variance of the machine learning method
The first step is to build several subsets of data from the training sample chosen at random with replacement, resulting in an ensemble of different models. Finally, the mean of predictions from different trees are considered. 
Remember that bagging takes all the features at every iteration.

### Random forest
Random forest is an extension of bagging, but taking one extra step. Along with taking the random subset of data, it also takes the random selection of features rather than using all features to grow trees. 
When you have many decision trees, it becomes a random forest
By allowing each tree to randomly sample from the dataset with replacement we get different trees since decision trees are sensitive to data they are trained on.

Random forest has many pros including accuracy, efficient runs on large databases, a large number of variables can be handled, 
it provides variable importance, missing data can be estimated, and accuracy is maintained when data is missing
Cons of random forest include occasional overfitting of data and biases over categorical variables with more levels.

Uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. 
The reason for this wonderful effect is that the trees protect each other from their individual errors. 
While some trees may be wrong, many other trees will be right, so as a group the trees can move in the correct direction. 
So the prerequisites for random forest to perform well are:

1. There needs to be some actual signal in our features so that models built using those features do better than random guessing.
2. The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.

Perform prediction from the created random forest classifier.


### The reason that the random forest model works so well

A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.
The low correlation between models is the key.

### Hyperparameters

A parameter of a model that is set before the start of the learning process is a hyperparameter. They can be adjusted manually.
Examples of hyperparameters include the number of nodes and layers in a neural network and the number of branches in a decision tree. 
Hyperparameters determine key features such as model architecture, learning rate, and model complexity.

Hyperparameters directly control model structure, function, and performance. 
Hyperparameter tuning allows data scientists to tweak model performance for optimal results. 
This process is an essential part of machine learning.


For example, assume you're using the learning rate of the model as a hyperparameter. If the value is too high, the model may converge too quickly with suboptimal results. Whereas if the rate is too low, training takes too long and results may not converge. A good and balanced choice of hyperparameters results in accurate models and excellent model performance.


Most used hyperparameters include:
* Number of trees
* Maximum depth of each tree
* Bootstrap method (sampling with/without replacement)
* Minimum data point needed to split at nodes, etc.

Validation curves and exhaustive grid search are the two techniques most commonly used to choose which hyperparameters to adjust. 
Validation curves visually check for potential values of hyperparameters which can be optimized. 
Exhaustive grid search tries every single possible combination of the hyperparameters. 
Remember when building validation curves, the other parameters should be held at their default values.

## Hyperparameter tuning techniques

To optimize the model’s performance it is important to tune the hyperparameters. 
There are three most widely used methods available such as 
1. Grid search, hyperparameter tuning chooses combinations of values from the range of categorical values that you specify when you create the job. Only categorical parameters are supported when using the grid search strategy. 
2. Random search, hyperparameter tuning chooses a random combination of hyperparameter values in the ranges that you specify for each training job it launches.
3. Bayesian optimization, treats hyperparameter tuning like a regression problem. Given a set of input features (the hyperparameters), hyperparameter tuning optimizes a model for the metric that you choose. 

These searches explore the different combinations of hyperparameter values that help to find the most effective configuration and fine-tune the decision tree model hyperparameter tuning makes guesses about which hyperparameter combinations are likely to get the best results. It then runs training jobs to test these values. After testing a set of hyperparameter values, hyperparameter tuning uses regression to choose the next set of hyperparameter values to test.

Common hyperparameters:

- Learning rate is the rate at which an algorithm updates estimates
- Learning rate decay is a gradual reduction in the learning rate over time to speed up learning
- Momentum is the direction of the next step with respect to the previous step
- Neural network nodes refers to the number of nodes in each hidden layer
- Mini-batch size is training data batch size
- Epochs is the number of times the entire training dataset is shown to the network during training
- Eta is step size shrinkage to prevent overfitting

**Note:** Decision forests,with default hyperparameters, in general will train quickly for small and medium sized problems.


## Model training


## 10 cents on AI  - Semianalysis by Google

Data quality scales better than data size
Save time by training on small, highly curated datasets. 
This suggests there is some flexibility in data scaling laws. 
The existence of such datasets follows from the line of thinking in Data Doesn’t Do What You Think, and they are rapidly 
becoming the standard way to do training. These datasets are built using synthetic methods (e.g. filtering the best responses from an existing model) and scavenging from other projects. 
Fortunately, these high quality datasets are open source.


## Fine Tuning ABC from Azure AI Foundry
Fine tuning adjusts the base model’s weights to improve performance on the specific task, 
you won’t have to include as many examples or instructions in your prompt. 
This means less text sent and fewer tokens processed on every API call.

Microsoft Azure AI Foundry 
-------------
Uses LoRA, or low rank approximation, to fine-tune models in a way that reduces their complexity without significantly affecting their performance. 
This method works by approximating the original high-rank matrix with a lower rank one, thus only fine-tuning a smaller subset of important parameters during the supervised training phase, 
making the model more manageable and efficient. For users, this makes training faster and more affordable than other techniques.

There are two unique fine-tuning experiences in the Azure AI Foundry portal:
Hub/Project view - supports fine-tuning models from multiple providers including Azure OpenAI, Meta Llama, Microsoft Phi, etc.
Azure OpenAI centric view - only supports fine-tuning Azure OpenAI models, but has support for additional features like the Weights & Biases (W&B) preview integration.

## How it works

At a high level, a machine algorithm creates one model data based on the existing test data as input.
Pushes the new input data then the machine learning algorithm makes a prediction based on the model which was prepared in step 1 above.
This prediction is evaluated and if accepted then an algorithm is deployed.
If the prediction is not accepted, then machine learning is trained again with bigger training data.