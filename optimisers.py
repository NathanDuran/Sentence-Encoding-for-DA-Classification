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
    optimisers = {'adadelta': 0.001,
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
        raise Exception("The given embedding processor type: '" + optimiser_type + "' is not valid!\n" +
                        "Please select one from: " + str(list(optimisers.keys())) + " or create one.")
    return optimiser


def adadelta(lr, clip_args, **kwargs):
    # Unpack key word arguments
    rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.95
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=epsilon, **clip_args)


def adagrad(lr, clip_args, **kwargs):
    # Unpack key word arguments
    initial_accum_value = kwargs['initial_accumulator_value'] if 'initial_accumulator_value' in kwargs.keys() else 0.1
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adagrad(learning_rate=lr, initial_accumulator_value=initial_accum_value,
                                       epsilon=epsilon, **clip_args)


def adam(lr, clip_args, **kwargs):
    # Unpack key word arguments
    beta_1 = kwargs['beta_1'] if 'beta_1' in kwargs.keys() else 0.9
    beta_2 = kwargs['beta_2'] if 'beta_2' in kwargs.keys() else 0.999
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    amsgrad = kwargs['amsgrad'] if 'amsgrad' in kwargs.keys() else False
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                    amsgrad=amsgrad, **clip_args)


def adamax(lr, clip_args, **kwargs):
    # Unpack key word arguments
    beta_1 = kwargs['beta_1'] if 'beta_1' in kwargs.keys() else 0.9
    beta_2 = kwargs['beta_2'] if 'beta_2' in kwargs.keys() else 0.999
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                      **clip_args)


def rmsprop(lr, clip_args, **kwargs):
    # Unpack key word arguments
    rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.9
    momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-07
    return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, epsilon=epsilon,
                                       **clip_args)


def sgd(lr, clip_args, **kwargs):
    # Unpack key word arguments
    momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
    nesterov = kwargs['nesterov'] if 'nesterov' in kwargs.keys() else False
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov, **clip_args)
