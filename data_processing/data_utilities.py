import pandas as pd


def save_dataframe(path, data, index_name='index'):
    data.to_csv(path, index_label=index_name)


def load_dataframe(path):
    data = pd.read_csv(path, index_col=False, header=0, quotechar="'")
    return data


def dataframe_wide_to_long(data):
    """Utility function for reshaping dataframes for plotting.
    Converts from 'wide' to 'long' format, where each observation is on a separate row.

    Example input:
              Multi-Pi
                 ap        da   ap type
    set 1  0.127465  0.404772  0.110144
    set 2  0.053677  0.252012  0.051167
    set 3  0.071018  0.367728  0.073543
    set 4  0.127416  0.465957  0.137216
    set 5  0.145793  0.441656  0.157172
    mean   0.105074  0.386425  0.105849

    Example output:
        index    metric group     value
    0   set 1  Multi-Pi    ap  0.127465
    1   set 2  Multi-Pi    ap  0.053677
    2   set 3  Multi-Pi    ap  0.071018
    3   set 4  Multi-Pi    ap  0.127416
    4   set 5  Multi-Pi    ap  0.145793
    5    mean  Multi-Pi    ap  0.105074
    ...etc
    """
    data = data.copy()
    data = data.reset_index().melt(id_vars=["index"])
    data = data.rename(columns={'variable_0': 'metric', 'variable_1': 'group'})
    data = data.dropna()
    return data


def sort_dataframe_by_list(data, sort_column, sort_order):

    # Create the dictionary that defines the order for sorting
    sorter_index = dict(zip(sort_order, range(len(sort_order))))

    # Generate a rank column that will be used to sort the dataframe numerically
    data['rank'] = data[sort_column].map(sorter_index)

    # Sort by rank and param
    data.sort_values(['rank'],  inplace=True)
    # Drop rank column
    data.drop('rank', 1, inplace=True)

    return data


def sort_dataframe_by_list_and_param(data, sort_column, sort_order, param):

    # Create the dictionary that defines the order for sorting
    sorter_index = dict(zip(sort_order, range(len(sort_order))))

    # Generate a rank column that will be used to sort the dataframe numerically
    data['rank'] = data[sort_column].map(sorter_index)

    # Sort by rank and param
    data.sort_values(['rank', param],  inplace=True)
    # Drop rank column
    data.drop('rank', 1, inplace=True)

    # Reset index
    data.reset_index(drop=True, inplace=True)

    return data


def get_means(data, experiment_type):
    """Creates dataframe of the means of each experiment value for an experiment_type."""
    # Create empty dataframe
    mean_data = pd.DataFrame(columns=data.columns)

    # Get the list of models and ranges of experiment values
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


def get_max(data, experiment_type):
    """Creates dataframe of the max validation/test/F1 and corresponding experiment value for an experiment_type."""
    # Get the index with the max validation accuracy by experiment_type value
    max_val = data.loc[data.groupby(['model_name'], sort=False)['val_acc'].idxmax()].reset_index()
    max_val.drop(max_val.columns.difference(['model_name', experiment_type, 'val_acc']), 1, inplace=True)
    max_val.rename(columns={experiment_type: 'val_' + experiment_type}, inplace=True)

    # Get the index with the max test/F1 accuracy by experiment_type value
    max_test = data.loc[data.groupby(['model_name'], sort=False)['test_acc'].idxmax()].reset_index()
    max_test.drop(max_test.columns.difference(['model_name', experiment_type, 'test_acc', 'f1_micro', 'f1_weighted']), 1, inplace=True)
    max_test.rename(columns={experiment_type: 'test_' + experiment_type}, inplace=True)

    # Group the validation and test data
    max_data = pd.concat([max_val, max_test], axis=1, ignore_index=False, sort=False)
    # Remove duplicate columns i.e. model_name
    max_data = max_data.loc[:, ~max_data.columns.duplicated()]
    # Remove '_' from column/model names
    max_data.columns = max_data.columns.str.replace("_", " ")
    # Round data to 6 decimals
    max_data = max_data.round(6)

    return max_data
