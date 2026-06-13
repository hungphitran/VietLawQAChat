# Training Strategy

The training pipeline has two main parts. The bi-encoder is trained with a ranking loss that encourages the query embedding to stay close to the positive passage embedding and far from negatives. The cross-encoder is trained on query-document pairs with positive and negative labels so that it can learn a more precise scoring function.

The current codebase already defines the training loops, hyperparameter configuration, and model saving policy. However, some training runs may still be in progress or may need final tuning before the report deadline. Those cases should be described as preliminary results rather than final conclusions. This keeps the report accurate while still documenting the intended training design.
