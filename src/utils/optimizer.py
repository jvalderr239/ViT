from functools import partial

from torch.optim.lr_scheduler import LambdaLR


def warmup(
    optimizer,
    training_steps: int,
    warmup_steps: int,
):
    """
    Linear warming rate scheduler

    Arguments:
        optimizer -- optimizer
        training_steps -- _description_
        warmup_steps -- _description_
    """
    def warmup_wrapper(
        current_step: int,
        training_steps: int,
        warmup_steps: int,
    ):
        """
        Wrapper for linear warming rate

        Arguments:
            current_step -- current step
            training_steps -- number of steps before scheduler stops
            warmup_steps -- number of steps before scheduler starts

        Returns:
            _description_
        """
        if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / warmup_steps)
        # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return max(
            0.0,
            float(training_steps - current_step)
            / float(max(1, training_steps - warmup_steps)),
        )

    lambda_warmup = partial(
        warmup_wrapper, training_steps=training_steps, warmup_steps=warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda=lambda_warmup)
