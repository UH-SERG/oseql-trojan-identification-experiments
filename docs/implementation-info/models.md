# `models.py` - Implementation Notes

## [Classification, Probabilities, and Loss Computation](https://github.com/UH-SERG/oseql-trojan-identification-experiments/blob/8209b7b88dafa733f2943a74caba72ce0bb5f3a1/models.py#L250-L258)

This code takes a feature vector `vec` and passes it through a classification layer to produce `logits`, which are raw, unnormalized scores for each class. These logits represent how strongly the model favors each class but are not yet probabilities.

The next line applies the softmax function to the logits to produce `prob`, which converts the raw scores into a probability distribution over classes where the values sum to one. This is typically used at inference time when you want interpretable class probabilities.

If labels are provided, the code assumes it is in training (or evaluation-with-labels) mode. It creates a cross-entropy loss function and computes the loss directly from the logits and the ground-truth labels. 

Importantly, `CrossEntropyLoss` internally applies `log_softmax`, so the loss is computed from the raw logits rather than the already-softmaxed probabilities. The function then returns both the loss and the probabilities.

If labels are not provided, the code assumes inference mode and returns only the predicted class probabilities.

