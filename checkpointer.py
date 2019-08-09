import os


class Checkpointer:
    """Class for saving a models checkpoints."""

    def __init__(self, checkpoint_dir, experiment_name, model, keep_best=1, minimise=True):
        """Constructs a Checkpointer for a given experiment and model.

        Can be used to save the best checkpoints during training according to a supplied metric value.

        Args:
            checkpoint_dir (str): The directory to save checkpoint files to
            experiment_name (str): Name of the experiment, used for creating checkpoint file names
            model (Model): Instance of the model to save, so its save function can be called
            keep_best (int): The number of 'best' checkpoints to keep
            minimise (bool): Whether the supplied metric values should be minimised (loss) or maximised (accuracy)

         Attributes:
             best_checkpoints (dict): Dictionary mapping checkpoint file names (keys) and the metric value (values)
        """

        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = model
        self.keep_best = keep_best
        self.minimise = minimise

        # Initialise the best checkpoints depending of the metric to monitor
        self.best_checkpoints = {}
        for i in range(self.keep_best):
            if self.minimise:
                self.best_checkpoints[str(i)] = float('inf')
            else:
                self.best_checkpoints[str(i)] = float('-inf')

    def save_best_checkpoint(self, metric_val, step):
        """Creates a new checkpoint if the current metric value is better than the least best in best_checkpoints dict.

        Keeps the number according to keep_best value.

        Args:
            metric_val (float): The current metric value to compare
            step (int): The current global step of the training model, used for creating checkpoint file names
        """

        # Sort the current best checkpoints according to the mode for the metric being monitored
        if self.minimise:
            sorted_checkpoints = sorted(self.best_checkpoints.items(), reverse=True, key=lambda kv: kv[1])
        else:
            sorted_checkpoints = sorted(self.best_checkpoints.items(), reverse=False, key=lambda kv: kv[1])
        # Get the least best key (checkpoint file name) and value (metric value)
        least_best_key = sorted_checkpoints[0][0]
        least_best_val = float(sorted_checkpoints[0][1])

        # Depending on mode, determine if new metric value is better
        if (self.minimise and metric_val <= least_best_val) or (not self.minimise and metric_val >= least_best_val):
            # Remove the checkpoint file if it exists
            if os.path.exists(os.path.join(self.checkpoint_dir, least_best_key)):
                os.remove(os.path.join(self.checkpoint_dir, least_best_key))
            # Remove from the best checkpoints dict
            self.best_checkpoints.pop(least_best_key)

            # Create a new checkpoint file
            ckpt_file = self.experiment_name + '_best_ckpt-{}.h5'.format(step)
            self.model.save_model(os.path.join(self.checkpoint_dir, ckpt_file))
            # Add to the best checkpoint dict
            self.best_checkpoints[ckpt_file] = metric_val

    def save_checkpoint(self, step):
        """Creates a new checkpoint file for the model at the current step.

        Args:
            step (int): The current global step of the training model, used for creating checkpoint file names
        """

        # Create a new checkpoint file
        ckpt_file = self.experiment_name + '_ckpt-{}.h5'.format(step)
        self.model.save_model(os.path.join(self.checkpoint_dir, ckpt_file))
