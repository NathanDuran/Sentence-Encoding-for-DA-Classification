import pandas as pd
from scipy.stats import shapiro, bartlett, levene, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


def shapiro_wilk_test(data, exp_param, metric):
    """Scipy Shapiro-Wilk test for normality.
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        shapiro_frame (Dataframe): Columns are model_name, exp_param, t-statistic and p-value.

            model_name vocab_size    t-stat   p-value
        0          cnn        500  0.933878  0.487099
        1          cnn       1000  0.934968  0.498493
        2          cnn       1500  0.904710  0.246593
    """
    # Create results frame
    shapiro_frame = pd.DataFrame(columns=['model_name', 'vocab_size', 't-stat', 'p-value'])

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    for model in model_names:

        # Create a list of all experiment values for this model
        model_data = data.loc[(data['model_name'] == model)]
        for exp_param_value in model_data[exp_param].unique():

            # Select the metric column
            metric_data = data.loc[(data['model_name'] == model) & (data[exp_param] == exp_param_value)][metric]

            # Run Shapiro-wilks
            t, p = shapiro(metric_data)

            # Append to result frame
            shapiro_frame = shapiro_frame.append({'model_name': model, 'vocab_size': exp_param_value,
                                                  't-stat': t, 'p-value': p}, ignore_index=True)

    return shapiro_frame


def levene_test(data, exp_param, metric):
    """Scipy Levene test for equal variance (Homoscedasticity).
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
    Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations
    from normality.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        levene_frame (Dataframe): Columns are model_name, t-statistic and p-value.

          model_name     t-stat   p-value
        0        cnn  22.431538  0.096978
        1   text cnn  13.273237  0.581202
        2       dcnn  13.045170  0.598809
    """
    # Create results frame
    levene_frame = pd.DataFrame(columns=['model_name', 't-stat', 'p-value'])

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    for model in model_names:

        # Create a list of all experiment values for this model
        model_data = data.loc[(data['model_name'] == model)]
        metric_data_list = []
        for exp_param_value in model_data[exp_param].unique():

            # Select the metric column
            metric_data_list.append(data.loc[(data['model_name'] == model) & (data[exp_param] == exp_param_value)][metric])

        # Run Levene's
        t, p = levene(*metric_data_list)

        # Append to result frame
        levene_frame = levene_frame.append({'model_name': model, 't-stat': t, 'p-value': p}, ignore_index=True)

    return levene_frame


def t_test(data, exp_param, metric):
    """Scipy T-test for two experiment groups.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        t_test_frame (Dataframe): Columns are t-statistic and p-value.

                 t-statistic   p-value
        0          2.598956  0.011838
    """
    # Get the list of models and ranges of experiment
    group_names = data[exp_param].unique()
    if len(group_names) != 2:
        raise ValueError("Too many groups in groups column! Found " + str(len(group_names)) + " but should == 2.")

    # Select the data to compare
    data_a = data.loc[(data[exp_param] == group_names[0])]
    data_b = data.loc[(data[exp_param] == group_names[1])]

    # T-test
    t_and_p = ttest_ind(data_a[metric], data_b[metric])

    # Create dataframe
    t_test_frame = pd.DataFrame({'t-statistic': [t_and_p[0]], 'p-value': [t_and_p[1]]})

    return t_test_frame


def anova_test(data, exp_param, metric):
    """ANOVA test with statsmodels.

    Eta-squared and omega-squared share the same suggested ranges for effect size classification:
    Low (0.01 – 0.059)
    Medium (0.06 – 0.139)
    Large (0.14+)
    Omega is considered a better measure of effect size than eta because it is unbiased in it’s calculation.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
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
        anova = ols(metric + '~ C(' + exp_param + ')', data=model_values_list).fit()
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


def tukey_hsd(data, exp_param, metric):
    """ANOVA and Tukey HSD post-hoc comparison from statsmodels.

    The Tukey HSD post-hoc comparison test controls for type I error and maintains the family-wise error rate at 0.05.
    The group1 and group2 columns are the groups being compared,
    meandiff column is the difference in means of the two groups being calculated as group2 – group1,
    lower/upper columns are the lower/upper boundaries of the 95% confidence interval,
    reject column states whether or not the null hypothesis should be rejected.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.

    Returns:
        tukey_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.

            model_name  group1  group2  meandiff  p-value   lower   upper  reject
        0          cnn     500    1000    0.0014   0.9000 -0.0021  0.0050   False
        1          cnn     500    1500    0.0041   0.0083  0.0006  0.0077    True
        2          cnn     500    2000    0.0051   0.0010  0.0016  0.0087    True
    """
    tukey_frame = pd.DataFrame()
    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    for model in model_names:
        # Create a list of all experiment values for this model
        model_data = data.loc[(data['model_name'] == model)]

        # Compare the results (metric) for the range of values for this experiment_type
        multi_comparison = MultiComparison(model_data[metric], model_data[exp_param])
        # Create the tukey results table
        tukey_results = multi_comparison.tukeyhsd()

        # Convert the results to a dataframe
        model_frame = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
        model_frame.insert(0, column='model_name', value=model)

        # Add to results frame
        tukey_frame = pd.concat([tukey_frame, model_frame], axis=0, ignore_index=True)

    tukey_frame.rename(columns={'p-adj': 'p-value'}, inplace=True)

    return tukey_frame
