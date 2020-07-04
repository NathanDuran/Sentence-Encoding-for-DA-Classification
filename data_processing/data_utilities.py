import os
import pandas as pd


def load_dataframe(path, multi_index=False, num_header_rows=1):
    # Create list for the number of rows with headers
    header_rows = [0 + i for i in range(num_header_rows + 1)]
    if multi_index:
        return pd.read_csv(path, header=header_rows, index_col=[0], skipinitialspace=True)
    else:
        return pd.read_csv(path, index_col=False, header=0, quotechar="'")


def save_dataframe(path, data, index_label='index'):
    # If the path doesn't exist make it first
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index_label=index_label)


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


def get_max(data, exp_param):
    """Creates dataframe of the max validation/test/F1 and corresponding experiment value for an experiment_type."""
    # Get the index with the max validation accuracy by experiment_type value
    max_val = data.loc[data.groupby(['model_name'], sort=False)['val_acc'].idxmax()].reset_index()
    max_val.drop(max_val.columns.difference(['model_name', exp_param, 'val_acc']), 1, inplace=True)
    max_val.rename(columns={exp_param: 'val_' + exp_param}, inplace=True)

    # Get the index with the max test/F1 accuracy by experiment_type value
    max_test = data.loc[data.groupby(['model_name'], sort=False)['test_acc'].idxmax()].reset_index()
    max_test.drop(max_test.columns.difference(['model_name', exp_param, 'test_acc', 'f1_micro']), 1, inplace=True)
    max_test.rename(columns={exp_param: 'test_' + exp_param}, inplace=True)

    # Group the validation and test data
    max_data = pd.concat([max_val, max_test], axis=1, ignore_index=False, sort=False)
    # Remove duplicate columns i.e. model_name
    max_data = max_data.loc[:, ~max_data.columns.duplicated()]
    # Round data to 6 decimals
    max_data = max_data.round(6)

    return max_data
