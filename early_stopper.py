class EarlyStopper:
    """Class for early stopping a models training."""

    def __init__(self, stopping=True, patience=1, min_delta=0.0, minimise=True):
        """Constructs a Checkpointer for a given experiment and model.

        Can be used to save the best checkpoints during training according to a supplied metric value.

        Args:
            stopping (bool): Whether to use early stopping or not
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            minimise (bool): Whether the supplied metric values should be minimised (loss) or maximised (accuracy)

        Attributes:
            last_improvement (int): Number of epochs since last improvement in metric
            current_best (float): Current best value for the monitored metric
        """

        self.stopping = stopping
        self.patience = patience
        self.min_delta = min_delta
        self.minimise = minimise
        self.last_improvement = 0
        # Initialise the current_best value depending on the metric to monitor
        if self.minimise:
            self.current_best = float('inf')
        else:
            self.current_best = float('-inf')

    def check_early_stop(self, metric_val):
        """Checks to see if training should stop early based on lack of improvement in supplied metric value.

        Args:
            metric_val (float): The current metric value to compare

        Returns:
            (bool): True if training should stop (no improvement for 'patience' number of epochs, else False
        """

        # If not using early stopping just return
        if not self.stopping:
            return False

        # Calculate the difference between current and best metric value
        delta = abs(metric_val - self.current_best)

        # Depending on mode, determine if new metric value is better (and delta is larger then min_delta)
        if ((self.minimise and metric_val <= self.current_best)
            or (not self.minimise and metric_val >= self.current_best)) \
                and (delta >= self.min_delta):

            # If it is set new best value and make sure counter is 0
            self.current_best = metric_val
            self.last_improvement = 0
            return False

        else:
            # Else increment epochs since last improvement
            self.last_improvement += 1

            # If num epochs since last improvement = patience then stop
            if self.last_improvement == self.patience:
                print("No improvement in monitored metric for " + str(self.patience) +
                      " epochs. Current: " + str(metric_val) + " vs Best: " + str(self.current_best) +
                      ". Stopping training.")
                return True
            else:
                return False
