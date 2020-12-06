import pandas as pd
from scipy.stats import shapiro, bartlett, levene, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


def shapiro_wilk_test(data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy Shapiro-Wilk test for normality.
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        shapiro_frame (Dataframe): Columns are model_name, exp_param, t-statistic and p-value.

            model_name vocab_size    t-stat   p-value
        0          cnn        500  0.933878  0.487099
        1          cnn       1000  0.934968  0.498493
        2          cnn       1500  0.904710  0.246593
    """

    # Create results frame
    shapiro_frame = pd.DataFrame(columns=['model_name', exp_param, 't-stat', 'p-value'])

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
            shapiro_frame = shapiro_frame.append({'model_name': model, exp_param: exp_param_value,
                                                  't-stat': t, 'p-value': p}, ignore_index=True)
    if show_result:
        if all(p_value > sig_level for p_value in shapiro_frame['p-value']):
            print("All models " + exp_param + " are normally distributed.")
        else:
            print("The following " + exp_param + " are not normally distributed.")
            print(shapiro_frame.loc[shapiro_frame['p-value'] <= sig_level])

    return shapiro_frame


def levene_test(data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy Levene test for equal variance (Homoscedasticity).
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
    Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations
    from normality.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        levene_frame (Dataframe): Columns are model_name, t-statistic and p-value.

          model_name     t-stat   p-value
        0        cnn  22.431538  0.096978
        1   text cnn  13.273237  0.581202
        2       dcnn  13.045170  0.598809
    """

    if exp_param != 'model_name':
        # Create results frame
        levene_frame = pd.DataFrame(columns=['model_name', exp_param, 't-stat', 'p-value'])

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

        if show_result:
            if all(p_value > sig_level for p_value in levene_frame['p-value']):
                print("All models " + exp_param + " have equal variance.")
            else:
                print("The following models " + exp_param + " do not have equal variance.")
                print(levene_frame.loc[levene_frame['p-value'] <= sig_level])

        return levene_frame
    else:
        levene_list = []
        for model in data['model_name'].unique():
            # Select the metric column
            levene_list.append(data.loc[(data['model_name'] == model)][metric])

        t, p_value = levene(*levene_list)

        if show_result:
            if p_value > sig_level:
                print("All models have equal variance. P-value = " + str(round(p_value, 5)))
            else:
                print("Some models do not have equal variance. P-value = " + str(round(p_value, 5)))


def t_test(data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy T-test for two experiment groups.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        t_test_frame (Dataframe): Columns are t-statistic and p-value.

                 t-statistic   p-value
        0          2.598956  0.011838
    """

    # Ensure only two groups are being tested
    if len(data[exp_param].unique()) != 2:
        raise ValueError("Too many groups in groups column! Found " + str(len(data[exp_param].unique())) + " but should == 2.")

    # Create results frame
    t_test_frame = pd.DataFrame(columns=['model_name', 't-stat', 'p-value'])

    # Get the list of models and experiment params
    model_names = data['model_name'].unique()
    groups = data[exp_param].unique()
    for model in model_names:

        # Select the data to compare
        data_a = data.loc[(data['model_name'] == model) & (data[exp_param] == groups[0])][metric]
        data_b = data.loc[(data['model_name'] == model) & (data[exp_param] == groups[1])][metric]

        # T-test
        t, p = ttest_ind(data_a, data_b)

        # Append to result frame
        t_test_frame = t_test_frame.append({'model_name': model, 't-stat': t, 'p-value': p}, ignore_index=True)

    if show_result:
        if all(p_value <= sig_level for p_value in t_test_frame['p-value']):
            print("All models have significant p-values when comparing " + exp_param + " groups.")
        else:
            print("The following models do not have significant p-values when comparing " + exp_param + " groups.")
            print(t_test_frame.loc[t_test_frame['p-value'] > sig_level])

    return t_test_frame


def one_way_anova_test(data, exp_param, metric, sig_level=0.05, show_result=True):
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
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

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

    if exp_param != 'model_name':
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
            #print(anova_table)

            # Add this models results to the dict
            results_dict[model] = anova_table.loc['C(' + exp_param + ')'].to_dict()

        # Create dataframe
        anova_frame = pd.DataFrame.from_dict(results_dict, orient='columns').T

        if show_result:
            if all(p_value <= sig_level for p_value in anova_frame['PR(>F)']):
                print("All models have significant p-values when comparing " + exp_param + " groups.")
            else:
                print("The following models do not have significant p-values when comparing " + exp_param + " groups.")
                print(anova_frame.loc[anova_frame['PR(>F)'] > sig_level])

        return anova_frame
    else:

        # Regression (ordinary least squares) for this metric and experiment type
        anova = ols(metric + '~ C(' + exp_param + ')', data=data).fit()
        # print(anova.summary())

        # Calculate the stats table
        anova_table = sm.stats.anova_lm(anova, typ=2)
        # Add effect size to table
        anova_table = _anova_table(anova_table)
        #print(anova_table)

        p_value = anova_table.iloc[0]['PR(>F)']

        if show_result:
            if p_value <= sig_level:
                print("The models have significant p-values when comparing groups. P-value = " + str(round(p_value, 5)))
            else:
                print("The models do not have significant p-values when comparing groups. P-value = " + str(round(p_value, 5)))

        return anova_table


def two_way_anova_test(data, exp_param1, exp_param2, metric, sig_level=0.05, show_result=True):
    """ANOVA test with statsmodels.

    Eta-squared and omega-squared share the same suggested ranges for effect size classification:
    Low (0.01 – 0.059)
    Medium (0.06 – 0.139)
    Large (0.14+)
    Omega is considered a better measure of effect size than eta because it is unbiased in it’s calculation.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param1 (string): Indicates first columns values to group data for comparison i.e. embedding_dim.
        exp_param2 (string): Indicates second columns values to group data for comparison i.e. embedding_type.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
    model_name                          exp_params    sum_sq   df   mean_sq         F    PR(>F)    eta_sq  omega_sq
            gru                    C(embedding_dim)  0.000023  4.0  0.000006  0.640506  0.634978  0.024733 -0.013749
            gru                   C(embedding_type)  0.000026  1.0  0.000026  2.944534  0.089609  0.028426  0.018592
            gru  C(embedding_dim):C(embedding_type)  0.000071  4.0  0.000018  2.020298  0.098257  0.078013  0.039022
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
    anova_frame = pd.DataFrame()
    for model in model_names:
        # Create a list of all experiment values for this model
        model_values_list = data.loc[(data['model_name'] == model)]

        # Regression (ordinary least squares) for this metric and experiment type
        formula = metric + '~ C(' + exp_param1 + ') + C(' + exp_param2 + ') + C(' + exp_param1 + '):C(' + exp_param2 + ')'
        anova = ols(formula, data=model_values_list).fit()
        # print(anova.summary())

        # Calculate the stats table
        anova_table = sm.stats.anova_lm(anova, typ=2)
        # Add effect size to table
        anova_table = _anova_table(anova_table)

        # Add model name column and append to results frame
        anova_table.index = anova_table.index.set_names(['exp_params'])
        anova_table.reset_index(inplace=True)
        anova_table.insert(0, 'model_name', model)
        anova_table = anova_table[:-1]  # Drop the residuals row
        # print(anova_table)

        anova_frame = pd.concat([anova_frame, anova_table], axis=0, ignore_index=True)

    if show_result:
        if all(p_value <= sig_level for p_value in anova_frame['PR(>F)']):
            print("All models have significant p-values when comparing " + exp_param1 + " and " + exp_param2 + " groups.")
        else:
            print("The following models do not have significant p-values when comparing " + exp_param1 + " and " + exp_param2 + " groups.")
            print(anova_frame.loc[anova_frame['PR(>F)'] > sig_level])

    return anova_frame


def tukey_hsd(data, exp_param, metric, sig_level=0.5, show_result=True):
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
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        tukey_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.

            model_name  group1  group2  meandiff  p-value   lower   upper  reject
        0          cnn     500    1000    0.0014   0.9000 -0.0021  0.0050   False
        1          cnn     500    1500    0.0041   0.0083  0.0006  0.0077    True
        2          cnn     500    2000    0.0051   0.0010  0.0016  0.0087    True
    """

    if exp_param != 'model_name':
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

        if show_result:
            if all(p_value <= sig_level for p_value in tukey_frame['p-value']):
                print("All models have significant p-values when comparing " + exp_param + " groups.")
            else:
                print("The following models do not have significant p-values when comparing " + exp_param + " groups.")
                print(tukey_frame.loc[tukey_frame['p-value'] > sig_level])

        return tukey_frame

    else:
        # Compare the results (metric) for the range of values for this experiment_type
        multi_comparison = MultiComparison(data[metric], data[exp_param])
        # Create the tukey results table
        tukey_results = multi_comparison.tukeyhsd()

        if show_result:
            print(tukey_results)
