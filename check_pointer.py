import os
from warnings import warn


class Checkpointer:
    """Class for saving a models checkpoints, weight files and embeddings."""

    def __init__(self, checkpoint_dir, experiment_name, model, save_ckpt=False, save_weights=False, keep_best=1, minimise=True):
        """Constructs a Checkpointer for a given experiment and model.

        Can be used to save the best checkpoints/weights during training according to a supplied metric value.

        Args:
            checkpoint_dir (str): The directory to save checkpoint files to
            experiment_name (str): Name of the experiment, used for creating checkpoint file names
            model (Model): Instance of the model to save, so its save function can be called
            save_ckpt (bool): Whether to save model checkpoints, if this and save weights is false nothing is saved
            save_weights (bool): Whether to save model weights
            keep_best (int): The number of 'best' checkpoints or weights to keep
            minimise (bool): Whether the supplied metric values should be minimised (loss) or maximised (accuracy)

         Attributes:
             best (dict): Dictionary mapping training steps (keys) and the metric value (values)
        """

        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = model
        self.save_ckpt = save_ckpt
        self.save_weights = save_weights
        self.keep_best = keep_best
        self.minimise = minimise

        # Display warning if neither save option is used
        if not self.save_ckpt and not self.save_weights:
            warn("Checkpointer save_ckpt and save_weights are False, nothing will be saved!")

        # Initialise the current best step dict depending of the metric to monitor
        self.best = {}
        for i in range(self.keep_best):
            if self.minimise:
                self.best['empty' + str(i)] = float('inf')
            else:
                self.best['empty' + str(i)] = float('-inf')

    def save_best(self, metric_val, step):
        """Creates a new checkpoint/weights file if the current metric value is better than the least best.

        Keeps the number according to keep_best value.

        Args:
            metric_val (float): The current metric value to compare
            step (int): The current global step of the training model, used for creating file names
        """
        if not self.save_ckpt and not self.save_weights:
            return

        # Sort the current best according to the mode for the metric being monitored
        if self.minimise:
            sorted_checkpoints = sorted(self.best.items(), reverse=True, key=lambda kv: kv[1])
        else:
            sorted_checkpoints = sorted(self.best.items(), reverse=False, key=lambda kv: kv[1])
        # Get the least best key (step) and value (metric value)
        least_best_key = sorted_checkpoints[0][0]
        least_best_val = float(sorted_checkpoints[0][1])

        # Depending on mode, determine if new metric value is better
        if (self.minimise and metric_val <= least_best_val) or (not self.minimise and metric_val >= least_best_val):

            # Remove the checkpoint and/or weights file if it exists
            ckpt_file = self.experiment_name + '-best-ckpt-{}.h5'.format(least_best_key)
            if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_file)):
                os.remove(os.path.join(self.checkpoint_dir, ckpt_file))

            weight_file = self.experiment_name + '-best-weights-{}.h5'.format(least_best_key)
            if os.path.exists(os.path.join(self.checkpoint_dir, weight_file)):
                os.remove(os.path.join(self.checkpoint_dir, weight_file))

            # Remove from 'best'
            self.best.pop(least_best_key)

            # Add to the 'best'
            self.best[str(step)] = metric_val

            # Create a new checkpoint file
            if self.save_ckpt:
                ckpt_file = self.experiment_name + '-best-ckpt-{}.h5'.format(step)
                self.model.save(os.path.join(self.checkpoint_dir, ckpt_file))

            # Create a new weights file
            if self.save_weights:
                weight_file = self.experiment_name + '-best-weights-{}.h5'.format(step)
                self.model.save_weights(os.path.join(self.checkpoint_dir, weight_file))

    def get_best(self):
        """Returns the key (step) with the best monitored metric from the current training session."""
        # Sort the current best according to the mode for the metric being monitored
        if self.minimise:
            sorted_checkpoints = sorted(self.best.items(), reverse=False, key=lambda kv: kv[1])
        else:
            sorted_checkpoints = sorted(self.best.items(), reverse=True, key=lambda kv: kv[1])

        # Return the best key (step) according to metric
        return sorted_checkpoints[0][0]

    def get_best_ckpt(self):
        """Returns the file name of the best checkpoint from the current training session, if it exists."""
        # Get the current best checkpoint
        best_key = self.get_best()

        # If this is a valid checkpoint file then return it
        ckpt_file = self.experiment_name + '-best-ckpt-{}.h5'.format(best_key)
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_file)):
            return os.path.join(self.checkpoint_dir, ckpt_file)
        else:
            return None

    def get_best_weights(self):
        """Returns the file name of the best weights from the current training session, if it exists."""
        # Get the current best checkpoint
        best_key = self.get_best()

        # If this is a valid checkpoint file then return it
        weight_file = self.experiment_name + '-best-weights-{}.h5'.format(best_key)
        if os.path.exists(os.path.join(self.checkpoint_dir, weight_file)):
            return os.path.join(self.checkpoint_dir, weight_file)
        else:
            return None

    def save(self, step):
        """Creates a new checkpoint and/or weight file for the model at the current step.

        Args:
            step (int): The current global step of the training model, used for creating file names
        """
        # Create a new checkpoint file
        if self.save_ckpt:
            ckpt_file = self.experiment_name + '-ckpt-{}.h5'.format(step)
            self.model.save(os.path.join(self.checkpoint_dir, ckpt_file))

        # Create a new weights file
        if self.save_weights:
            weight_file = self.experiment_name + '-weights-{}.h5'.format(step)
            self.model.save_weights(os.path.join(self.checkpoint_dir, weight_file))

    def save_embeddings(self, output_dir, vocabulary, layer_name='embedding'):
        """Creates a word embedding .txt file from the models embedding layer.

        Args:
            output_dir (str): Location to save the embedding file
            vocabulary (Gluonnlp Vocab): Data sets vocabulary for mapping indexes to words
            layer_name (str): Name of the embedding layer, default = 'embedding'
        """
        # Get the embedding weights from the model layer
        embedding_weights = self.model.get_layer(name=layer_name).get_weights()[0]

        # If embeddings are not trained the layer will not have weights
        if embedding_weights:
            with open(os.path.join(output_dir, self.experiment_name + '-{:03d}d.txt'.format(embedding_weights.shape[1])), 'w') as file:
                for i in range(embedding_weights.shape[0]):  # Vocab size
                    line = vocabulary.idx_to_token[i] + " "
                    for j in range(embedding_weights.shape[1]):  # Embedding dim
                        line += str(embedding_weights[i][j]) + " "
                    line += "\n"
                    file.write(line)
        else:
            print("Embedding weights are 'NoneType', for character models this is correct, "
                  "otherwise make sure train_embeddings=True.")
