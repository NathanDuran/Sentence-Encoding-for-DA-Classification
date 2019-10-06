import os


class Checkpointer:
    """Class for saving a models checkpoints."""

    def __init__(self, checkpoint_dir, experiment_name, model, saving=True, save_weights=False, keep_best=1, minimise=True):
        """Constructs a Checkpointer for a given experiment and model.

        Can be used to save the best checkpoints during training according to a supplied metric value.

        Args:
            checkpoint_dir (str): The directory to save checkpoint files to
            experiment_name (str): Name of the experiment, used for creating checkpoint file names
            model (Model): Instance of the model to save, so its save function can be called
            saving (bool): Whether to actually save models i.e. disables checkpointer
            save_weights (bool): Whether to save weights as well as model
            keep_best (int): The number of 'best' checkpoints to keep
            minimise (bool): Whether the supplied metric values should be minimised (loss) or maximised (accuracy)

         Attributes:
             best_checkpoints (dict): Dictionary mapping checkpoint file names (keys) and the metric value (values)
        """

        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = model
        self.saving = saving
        self.save_weights = save_weights
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
        if self.saving:
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
                ckpt_file = self.experiment_name + '-best-ckpt-{}.h5'.format(step)
                self.model.save(os.path.join(self.checkpoint_dir, ckpt_file))
                # Add to the best checkpoint dict
                self.best_checkpoints[ckpt_file] = metric_val

                # Check if saving weights as well
                if self.save_weights:
                    weight_file = self.experiment_name + '-best-weights-{}.h5'.format(step)
                    self.model.save_weights(os.path.join(self.checkpoint_dir, weight_file))

    def save_checkpoint(self, step):
        """Creates a new checkpoint file for the model at the current step.

        Args:
            step (int): The current global step of the training model, used for creating checkpoint file names
        """
        if self.saving:
            # Create a new checkpoint file
            ckpt_file = self.experiment_name + '-ckpt-{}.h5'.format(step)
            self.model.save(os.path.join(self.checkpoint_dir, ckpt_file))

            # Check if saving weights as well
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
