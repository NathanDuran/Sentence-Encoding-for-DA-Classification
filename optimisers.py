import tensorflow as tf


def get_optimiser(optimiser_type='sgd', lr=0.01, **kwargs):
    """Utility function for returning a Keras optimiser.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

    If no arguments provided then defaults to SGD with learning rate of 0.01.

    Args:
        optimiser_type (str): The name of the optimisation algorithm, must be in the optimisers dict
        lr (float): The learning rate to use
        kwargs (dict): Can contain any keyword arguments for the keras optimisers, otherwise uses default values

    Returns:
         optimiser (tf.keras.optimizers.Optimizer): An instance of a keras optimiser
    """

    # List of valid optimisers and default learning rates
    optimisers = {'adadelta': 1.0,
                  'adagrad': 0.001,
                  'adam': 0.001,
                  'adamax': 0.001,
                  'rmsprop': 0.001,
                  'sgd': 0.01}

    # Check an optimiser has been specified in the kwargs
    if optimiser_type.lower() in optimisers.keys():

        # Get gradient clipping values if they exist
        clip_args = {}
        if 'clipnorm' in kwargs.keys():
            clip_args['clipnorm'] = kwargs['clipnorm']
        if 'clipvalue' in kwargs.keys():
            clip_args['clipvalue'] = kwargs['clipvalue']

        # Build the optimiser with the specified params
        optimiser = globals()[optimiser_type.lower()](lr, clip_args, **kwargs)
    else:
        raise Exception("The given optimiser type: '" + optimiser_type + "' is not valid!\n" +
                        "Please select one from: " + str(list(optimisers.keys())) + " or create one.")
    return optimiser


def adadelta(lr, clip_args, **kwargs):
    """Adadelta optimization is a stochastic gradient descent method that is based on adaptive learning rate per
     dimension to address two drawbacks:
     1) the continual decay of learning rates throughout training
     2) the need for a manually selected global learning rate

    Two accumulation steps are required:
    1) the accumulation of gradients squared,
    2) the accumulation of updates squared.

    Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient
    updates, instead of accumulating all past gradients.
    This way, Adadelta continues learning even when many updates have been done.
    Compared to Adagrad, in the original version of Adadelta you don't have to set an initial learning rate.
    In this version, initial learning rate can be set, as in most other Keras optimizers.
    """
    rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.95
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=epsilon, **clip_args)


def adagrad(lr, clip_args, **kwargs):
    """Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently
     a parameter gets updated during training. The more updates a parameter receives, the smaller the updates.
     """
    initial_accum_value = kwargs['initial_accumulator_value'] if 'initial_accumulator_value' in kwargs.keys() else 0.1
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adagrad(learning_rate=lr, initial_accumulator_value=initial_accum_value,
                                       epsilon=epsilon, **clip_args)


def adam(lr, clip_args, **kwargs):
    """Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order
     and second-order moments. According to the paper Adam: A Method for Stochastic Optimization.Kingma et al., 2014,
     the method is "computationally efficient, has little memory requirement,invariant to diagonal rescaling of
      gradients, and is well suited for problems that are large in terms of data/parameters".
    """
    beta_1 = kwargs['beta_1'] if 'beta_1' in kwargs.keys() else 0.9
    beta_2 = kwargs['beta_2'] if 'beta_2' in kwargs.keys() else 0.999
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    amsgrad = kwargs['amsgrad'] if 'amsgrad' in kwargs.keys() else False
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                    amsgrad=amsgrad, **clip_args)


def adamax(lr, clip_args, **kwargs):
    """It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper.
    Adamax is sometimes superior to adam, specially in models with embeddings.
    """
    beta_1 = kwargs['beta_1'] if 'beta_1' in kwargs.keys() else 0.9
    beta_2 = kwargs['beta_2'] if 'beta_2' in kwargs.keys() else 0.999
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                      **clip_args)


def rmsprop(lr, clip_args, **kwargs):
    """This optimizer is usually a good choice for recurrent neural networks."""
    rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.9
    momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, epsilon=epsilon,
                                       **clip_args)


def sgd(lr, clip_args, **kwargs):
    momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
    nesterov = kwargs['nesterov'] if 'nesterov' in kwargs.keys() else False
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov, **clip_args)
