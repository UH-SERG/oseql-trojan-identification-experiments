# `run_defect.py` - Implementation Notes

## [AdamW Optimizer Initialization](https://github.com/UH-SERG/oseql-trojan-identification-experiments/blob/8209b7b88dafa733f2943a74caba72ce0bb5f3a1/run_defect.py#L317)
This line creates an optimizer using the AdamW algorithm, which updates model parameters during training. 

`optimizer_grouped_parameters` specifies which parameters are optimized and allows different settings (like weight decay) for different parameter groups. 

The learning rate `lr=args.learning_rate` controls how large each update step is, and `eps=args.adam_epsilon` is a small constant added for numerical stability to avoid division by very small numbers. 

AdamW is preferred over Adam because it applies weight decay correctly, leading to better regularization and generalization.

## [Learning Rate Warmup and Scheduler Setup](https://github.com/UH-SERG/oseql-trojan-identification-experiments/blob/8209b7b88dafa733f2943a74caba72ce0bb5f3a1/run_defect.py#L323-L324)

This code computes how many warmup steps to use for the learning rate schedule and then creates a scheduler. 

If `args.warmup_steps` is less than 1, it is treated as a fraction of the total number of training steps, so the actual warmup steps are computed as that fraction of `num_train_optimization_steps`. If it is 1 or greater, it is treated as an absolute number of warmup steps and explicitly converted to an integer.

The `scheduler` is then created using a linear learning rate schedule with warmup (`get_linear_schedule_with_warmup` from the `transformers` library). 

According to the above scheduler, during the warmup phase, the learning rate increases linearly from zero to the target learning rate, and after warmup it decreases linearly to zero over the remaining training steps. 

The scheduler is tied to the optimizer and will be stepped during training to control how the learning rate evolves. 

[Calling `scheduler.step()`](https://github.com/UH-SERG/oseql-trojan-identification-experiments/blob/8209b7b88dafa733f2943a74caba72ce0bb5f3a1/run_defect.py#L364) advances the scheduler by one step and updates the learning rate according to the schedule (warmup phase or decay phase). If you don’t “step” the scheduler, the learning rate will stay at its initial value.
