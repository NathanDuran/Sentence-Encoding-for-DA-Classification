import os
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

def save_dataframe(path, data):
    data.to_csv(path, index_label='labels')


def load_experiment_data(task_name, experiment_type):
    data_path = os.path.join('..', task_name, task_name + '_' + experiment_type + '.csv')
    data = pd.read_csv(data_path, index_col=False, header=0, quotechar="'")
    data = data.drop('experiment_name', axis='columns')
    data.model_name = data.model_name.str.replace("_", " ")
    return data


def sort_experiment_data_by_model(data):
    # Model sort order
    sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'bi lstm', 'gru', 'bi gru']

    # Create the dictionary that defines the order for sorting
    sorter_index = dict(zip(sort_order, range(len(sort_order))))

    # Generate a rank column that will be used to sort the dataframe numerically
    data['model_name_rank'] = data['model_name'].map(sorter_index)

    # Sort by model name (rank) and metric
    data.sort_values(['model_name_rank'],  inplace=True)
    # Drop rank column
    data.drop('model_name_rank', 1, inplace=True)

    return data


def sort_experiment_data_by_model_and_metric(data, metric):
    # Model sort order
    sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'bi lstm', 'gru', 'bi gru']

    # Create the dictionary that defines the order for sorting
    sorter_index = dict(zip(sort_order, range(len(sort_order))))

    # Generate a rank column that will be used to sort the dataframe numerically
    data['model_name_rank'] = data['model_name'].map(sorter_index)

    # Sort by model name (rank) and metric
    data.sort_values(['model_name_rank', metric],  inplace=True)
    # Drop rank column
    data.drop('model_name_rank', 1, inplace=True)

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


def t_test(data, experiment_type, metric):

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    results_dict = dict()
    for model in model_names:
        model_dict = dict()
        for i in range(len(experiment_values) - 1):

            # Select the data to compare
            data_a = data.loc[(data['model_name'] == model) & (data[experiment_type] == experiment_values[i])]
            data_b = data.loc[(data['model_name'] == model) & (data[experiment_type] == experiment_values[i+1])]

            # T-test
            t_and_p = ttest_ind(data_a[metric], data_b[metric])
            # Add the p value to this pair in the dict
            model_dict[str(experiment_values[i]) + " & " + str(experiment_values[i+1])] = t_and_p[1]

        # Add this models results to the dict
        results_dict[model] = model_dict

    # Create dataframe
    t_test_frame = pd.DataFrame.from_dict(results_dict, orient='index').reindex(results_dict.keys())

    return t_test_frame


def anova_test(data, experiment_type, metric):

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    results_dict = dict()
    for model in model_names:

        # Create a list of all experiment values for this model and metric
        model_values_list = [data.loc[(data['model_name'] == model) & (data[experiment_type] == val)][metric].tolist() for val in experiment_values]

        # Unpack list and generate stats
        anova = f_oneway(*model_values_list)

        # Add this models results to the dict
        results_dict[model] = anova

    # Create dataframe
    anova_test_frame = pd.DataFrame.from_dict(results_dict, orient='index')

    # Round to 4 decimal places
    anova_test_frame = anova_test_frame.round(6)
    return anova_test_frame


def anova_test2(data, experiment_type, metric):

    def _anova_table(aov):
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    results_dict = dict()
    for model in model_names:

        # Create a list of all experiment values for this model and metric
        model_values_list = data.loc[(data['model_name'] == model)]

        # Unpack list and generate stats
        anova = ols(metric + '~ C(' + experiment_type + ')', data=model_values_list).fit()
        # print(anova.summary())
        aov_table = sm.stats.anova_lm(anova, typ=2)
        print(_anova_table(aov_table))



    #     # Add this models results to the dict
    #     results_dict[model] = anova
    #
    # # Create dataframe
    # anova_test_frame = pd.DataFrame.from_dict(results_dict, orient='index')
    #
    # # Round to 4 decimal places
    # anova_test_frame = anova_test_frame.round(6)
    # return anova_test_frame
