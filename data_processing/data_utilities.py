import os
import pandas as pd


def save_dataframe(path, data):
    data.to_csv(path, index_label='labels')


def load_experiment_data(task_name, experiment_type):
    data_path = os.path.join('..', task_name, task_name + '_' + experiment_type + '.csv')
    data = pd.read_csv(data_path, index_col=False, header=0, quotechar="'")
    data = data.drop('experiment_name', axis='columns')
    return data


def get_experiment_means(data, experiment_type):

    # Create empty dataframe
    mean_data = pd.DataFrame(columns=data.columns)

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    for model in model_names:
        for exp_value in experiment_values:
            # Select only the data we want
            model_data = data.loc[(data['model_name'] == model) & (data[experiment_type] == exp_value)]
            # Get the first lines metadata as series
            meta = model_data.loc[:, 'model_name': 'embedding_source'].iloc[0]
            # Get the mean of the metrics
            mean = model_data.loc[:, 'train_loss':].mean()
            # Create a dataframe and append to overall results
            model_data = pd.concat([meta, mean], axis=0)
            mean_data = mean_data.append(model_data, ignore_index=True)

    return mean_data