import pandas as pd
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


def pairwise_t_test(data, experiment_type, metric):
    """Pairwise t-test for each consecutive experiment value.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        experiment_type (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        t_test_frame (Dataframe): Columns are experiment_type value pairs, rows are model_name and data is the p-value.

                  500 & 1000  1000 & 1500  1500 & 2000  2000 & 2500  2500 & 3000  3000 & 3500
        cnn         0.000483     0.040621     0.546059     0.274387     0.418656     0.647783
        text cnn    0.000330     0.545221     0.625784     0.203548     0.243866     0.035153
        dcnn        0.000693     0.094072     0.951321     0.810791     0.812316     0.225113
    """
    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    results_dict = dict()
    for model in model_names:
        model_dict = dict()
        for i in range(len(experiment_values) - 1):
            # Select the data to compare
            data_a = data.loc[(data['model_name'] == model) & (data[experiment_type] == experiment_values[i])]
            data_b = data.loc[(data['model_name'] == model) & (data[experiment_type] == experiment_values[i + 1])]

            # T-test
            t_and_p = ttest_ind(data_a[metric], data_b[metric])
            # Add the p value to this pair in the dict
            model_dict[str(experiment_values[i]) + " & " + str(experiment_values[i + 1])] = t_and_p[1]

        # Add this models results to the dict
        results_dict[model] = model_dict

    # Create dataframe
    t_test_frame = pd.DataFrame.from_dict(results_dict, orient='index').reindex(results_dict.keys())

    return t_test_frame


def f_oneway_test(data, experiment_type, metric):
    """ANOVA test with scipy.stats.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        experiment_type (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        f_one_way_frame (Dataframe): Columns are f-statistic and p-value value pairs and rows are model_name.

                   statistic pvalue
        cnn        5.055341  0.000000
        text cnn   4.298028  0.000001
        dcnn       3.555700  0.000032
    """
    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    experiment_values = data[experiment_type].unique()
    results_dict = dict()
    for model in model_names:
        # Create a list of all experiment values for this model and metric
        model_values_list = [data.loc[(data['model_name'] == model) & (data[experiment_type] == val)][metric].tolist()
                             for val in experiment_values]

        # Unpack list and generate stats
        model_result = f_oneway(*model_values_list)

        # Add this models results to the dict
        results_dict[model] = model_result

    # Create dataframe
    f_one_way_frame = pd.DataFrame.from_dict(results_dict, orient='index')
    # Round to 6 decimal places
    f_one_way_frame = f_one_way_frame.round(6)
    return f_one_way_frame


def anova_test(data, experiment_type, metric):
    """ANOVA test with statsmodels.

    Eta-squared and omega-squared share the same suggested ranges for effect size classification:
    Low (0.01 – 0.059)
    Medium (0.06 – 0.139)
    Large (0.14+)
    Omega is considered a better measure of effect size than eta because it is unbiased in it’s calculation.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        experiment_type (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.

                         F    PR(>F)    df    eta_sq   mean_sq  omega_sq    sum_sq
        cnn       5.055341  0.000000  15.0  0.344949  0.000139  0.275461  0.002084
        text cnn  4.298028  0.000001  15.0  0.309255  0.000080  0.236169  0.001198
        dcnn      3.555700  0.000032  15.0  0.270278  0.000102  0.193286  0.001528
    """

    def _anova_table(aov_data):
        """Calculates the effect size statisctics."""
        aov_data['mean_sq'] = aov_data[:]['sum_sq'] / aov_data[:]['df']
        aov_data['eta_sq'] = aov_data[:-1]['sum_sq'] / sum(aov_data['sum_sq'])
        aov_data['omega_sq'] = (aov_data[:-1]['sum_sq'] - (aov_data[:-1]['df'] * aov_data['mean_sq'][-1])) / (sum(aov_data['sum_sq']) + aov_data['mean_sq'][-1])
        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov_data = aov_data[cols]
        return aov_data

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    results_dict = dict()
    for model in model_names:
        # Create a list of all experiment values for this model
        model_values_list = data.loc[(data['model_name'] == model)]

        # Regression (ordinary least squares) for this metric and experiment type
        anova = ols(metric + '~ C(' + experiment_type + ')', data=model_values_list).fit()
        # print(anova.summary())

        # Calculate the stats table
        anova_table = sm.stats.anova_lm(anova, typ=2)
        # Add effect size to table
        anova_table = _anova_table(anova_table)

        # Add this models results to the dict
        results_dict[model] = anova_table.loc['C(vocab_size)'].to_dict()

    # Create dataframe
    anova_frame = pd.DataFrame.from_dict(results_dict, orient='columns').T
    # Round to 6 decimal places
    anova_frame = anova_frame.round(6)
    return anova_frame


def tukey_hsd(data, experiment_type, metric):
    """ANOVA and Tukey HSD post-hoc comparison from statsmodels.

    The Tukey HSD post-hoc comparison test controls for type I error and maintains the familywise error rate at 0.05.
    The group1 and group2 columns are the groups being compared,
    meandiff column is the difference in means of the two groups being calculated as group2 – group1,
    lower/upper columns are the lower/upper boundaries of the 95% confidence interval,
    reject column states whether or not the null hypothesis should be rejected.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        experiment_type (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        tukey_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.

            model_name  group1  group2  meandiff  p-value   lower   upper  reject
        0          cnn     500    1000    0.0014   0.9000 -0.0021  0.0050   False
        1          cnn     500    1500    0.0041   0.0083  0.0006  0.0077    True
        2          cnn     500    2000    0.0051   0.0010  0.0016  0.0087    True
    """
    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    tukey_frame = pd.DataFrame()
    for model in model_names:
        # Create a list of all experiment values for this model
        model_data = data.loc[(data['model_name'] == model)]

        # Compare the results (metric) for the range of values for this experiment_type
        multi_comparison = MultiComparison(model_data[metric], model_data[experiment_type])
        # Create the tukey results table
        tukey_results = multi_comparison.tukeyhsd()

        # Convert the results to a dataframe
        model_frame = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
        model_frame.insert(0, column='model_name', value=model)

        # Add to results frame
        tukey_frame = pd.concat([tukey_frame, model_frame], axis=0, ignore_index=True)

    tukey_frame.rename(columns={'p-adj': 'p-value'}, inplace=True)

    return tukey_frame
