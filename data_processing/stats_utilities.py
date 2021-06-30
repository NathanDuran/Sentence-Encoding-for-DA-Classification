import os
import pandas as pd
import numpy as np
from math import sqrt
import itertools
from data_utilities import load_predictions
import bayesian_tests as bt
import pingouin as pg
from statistics import variance, mean
from scipy.stats import shapiro, bartlett, levene, ttest_ind, ttest_rel, wilcoxon
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.oneway import effectsize_oneway
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def _add_subjects(x, num=10):
    """Adds a 'subjects' column to dataframe."""
    x['subject'] = [i for i in range(1, num+1)]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return x


def shapiro_wilk_test(input_data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy Shapiro-Wilk test for normality.
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

    Args:
        input_data (Dataframe): Dataframe grouped by model_name and experiment_type values.
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
    data = input_data.copy(deep=True)

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


def levene_test(input_data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy Levene test for equal variance (Homoscedasticity).
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
    Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations
    from normality.

          model_name     t-stat   p-value
        0        cnn  22.431538  0.096978
        1   text cnn  13.273237  0.581202
        2       dcnn  13.045170  0.598809

    Args:
        input_data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        levene_frame (Dataframe): Columns are model_name, t-statistic and p-value.
    """
    data = input_data.copy(deep=True)

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
                print("Some models do not have equal variance. " + str(data['model_name'].unique()) + " P-value = " + str(round(p_value, 5)))


def mauchly_test(input_data, exp_param, metric, sig_level=0.05, show_result=True):
    """Pingouin Mauchly and JNS test for sphericity.

    Args:
        input_data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        mauch_frame (Dataframe): Columns are model_name, t-statistic and p-value.
    """
    data = input_data.copy(deep=True)

    # Add 'subjects' to all models
    data = data.groupby(['model_name', exp_param]).apply(_add_subjects)

    # Create results frame
    mauch_frame = pd.DataFrame(columns=['model_name', exp_param, 'sphere', 'W', 'chisq', 'p-value'])

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    for model in model_names:

        # Create a list of all experiment values for this model
        model_data = data.loc[(data['model_name'] == model)]

        # Run Mauchley's
        spher, W, chisq, dof, pval = pg.sphericity(model_data, dv=metric, within=exp_param, subject='subject')

        # Append to result frame
        mauch_frame = mauch_frame.append({'model_name': model, 'sphere': spher, 'W': W, 'chisq':chisq, 'p-value': pval}, ignore_index=True)

    if show_result:
        if all(p_value > sig_level for p_value in mauch_frame['p-value']):
            print("All models " + exp_param + " sphericity assumption is met.")
        else:
            print("The following models " + exp_param + " sphericity assumption is not met.")
            print(mauch_frame.loc[mauch_frame['p-value'] <= sig_level])

        return mauch_frame


def cohen_d(data_a, data_b):
    """"Cohens d effect size, for calculating the difference between the mean value of groups.

    Args:
        data_a (list): One group of samples.
        data_b (list): One group of samples.

    Returns:
        cohens_d (float): Cohens d effect size.
    """
    # Calculate the size of samples
    n1, n2 = len(data_a), len(data_b)

    # Calculate the variance of the samples
    s1, s2 = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

    # Calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # Calculate the means of the samples
    u1, u2 = np.mean(data_a), np.mean(data_b)

    # Calculate the effect size
    return (u1 - u2) / s


def coehen_f(data, groups, metric):
    """"Cohens f, for calculating the effect size for anova power analysis.

    Args:
        data (DataFrame): DataFrame with columns 'groups' and 'metric'.
        groups (str): Name of the groups column in the data, i.e. the groups for the anova test.
        metric (str): Name of the metric column in the data, i.e. the dependant variable.

    Returns:
        cohens_f (float): Cohens f effect size.
    """
    # Get the means and variance of the data
    means = data.groupby([groups], sort=False).apply(lambda x: mean(x[metric])).to_list()
    variances = data.groupby([groups], sort=False).apply(lambda x: variance(x[metric])).to_list()

    # Calculate effect size
    cohens_f = effectsize_oneway(means, variances, len(data), use_var='equal')
    return cohens_f


def t_test_power_analysis(data_a, data_b, alpha=0.05, power=0.8):
    """Performs statistical power analysis for t-test.
    First calculates cohens d effect size,
    then calculates an expected number of samples for given power, and the actual power given the actual sample size.

    Args:
        data_a (list): One group of samples.
        data_b (list): One group of samples.
        alpha (float): Significance level.
        power (float): Desired statistical power for calculating the expected num samples.

    Returns:
        cohens_d (float): Cohens d effect size.
        exp_n_samples (float): The expected number of samples given desired power, effect size and alpha.
        act_power (float): The actual power given the actual n_samples, effect size and alpha.
    """
    # Parameters for power analysis
    effect = cohen_d(data_a, data_b)  # Cohens d effect size
    n_samples = len(data_a)

    # perform power analysis
    analysis = TTestIndPower()
    exp_n_samples = analysis.solve_power(effect_size=effect, power=power, ratio=1.0, alpha=alpha)
    # print('Expected Sample Size: %.3f' % exp_n_samples)

    act_power = analysis.solve_power(effect_size=effect, nobs1=n_samples, ratio=1.0, alpha=alpha)
    # print('Actual Power: %.3f' % act_power)
    return effect, exp_n_samples, act_power


def anova_power_analysis(data, groups, metric, alpha=0.05, power=0.8):
    """Performs statistical power analysis for t-test.
    First calculates cohens d effect size,
    then calculates an expected number of samples for given power, and the actual power given the actual sample size.

    Args:
        data (DataFrame): DataFrame with columns 'groups' and 'metric'.
        groups (str): Name of the groups column in the data, i.e. the groups for the anova test.
        metric (str): Name of the metric column in the data, i.e. the dependant variable.
        alpha (float): Significance level.
        power (float): Desired statistical power for calculating the expected num samples.

    Returns:
        cohens_f (float): Cohens f effect size.
        exp_n_samples (float): The expected number of samples given desired power, effect size and alpha.
        act_power (float): The actual power given the actual n_samples, effect size and alpha.
    """
    # Calculate Cohen's f effect size
    effect = coehen_f(data, groups, metric)
    # FTestAnovaPower() cannot handle large effect sizes so just manually set
    if effect > 1.4:
        effect = 1.4

    # Number of samples
    n_samples = len(data)
    n_groups = len(data[groups].unique())

    # Expected num samples
    exp_n_samples = FTestAnovaPower().solve_power(effect_size=sqrt(effect), alpha=alpha, power=power, k_groups=n_groups)
    # print('Expected Sample Size: %.3f' % exp_n_samples)
    act_power = FTestAnovaPower().solve_power(effect_size=sqrt(effect), nobs=n_samples, alpha=alpha, k_groups=n_groups)
    # print('Actual Power: %.3f' % act_power)

    return effect, exp_n_samples, act_power


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

        # Perform power analysis
        effect, exp_n, act_power = t_test_power_analysis(data_a, data_b, alpha=sig_level, power=0.8)

        # T-test
        t, p = ttest_rel(data_a, data_b)

        # Append to result frame
        t_test_frame = t_test_frame.append({'model_name': model, 't-stat': t, 'p-value': p, 'cohen-d': effect,
                                            'n': len(data_a), 'exp_n': exp_n, 'power': act_power, 'exp_power': 0.8},
                                           ignore_index=True)

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

                        F    PR(>F)    df    eta_sq   mean_sq  omega_sq    sum_sq
        cnn       5.055341  0.000000  15.0  0.344949  0.000139  0.275461  0.002084
        text cnn  4.298028  0.000001  15.0  0.309255  0.000080  0.236169  0.001198
        dcnn      3.555700  0.000032  15.0  0.270278  0.000102  0.193286  0.001528

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
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

            # Add power analysis
            effect, exp_n, power = anova_power_analysis(model_values_list, exp_param, metric, alpha=sig_level, power=0.8)

            # Add this models results to the dict
            results_dict[model] = anova_table.loc['C(' + exp_param + ')'].to_dict()
            power_analysis_cols = {'cohen_f': effect, 'n': len(model_values_list), 'exp_n': exp_n, 'power': power, 'exp_power': 0.8}
            results_dict[model] = {**results_dict[model], **power_analysis_cols}

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

        # Add power analysis
        effect, exp_n, power = anova_power_analysis(data, exp_param, metric, alpha=sig_level, power=0.8)
        power_analysis_cols = {'cohen_f': effect, 'n': len(data), 'exp_n': exp_n, 'power': power, 'exp_power': 0.8}
        anova_table = anova_table.assign(**power_analysis_cols)

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

    model_name                          exp_params    sum_sq   df   mean_sq         F    PR(>F)    eta_sq  omega_sq
            gru                    C(embedding_dim)  0.000023  4.0  0.000006  0.640506  0.634978  0.024733 -0.013749
            gru                   C(embedding_type)  0.000026  1.0  0.000026  2.944534  0.089609  0.028426  0.018592
            gru  C(embedding_dim):C(embedding_type)  0.000071  4.0  0.000018  2.020298  0.098257  0.078013  0.039022

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param1 (string): Indicates first columns values to group data for comparison i.e. embedding_dim.
        exp_param2 (string): Indicates second columns values to group data for comparison i.e. embedding_type.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
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


def rm_one_way_anova_test(data, exp_param, metric, sig_level=0.05, show_result=True):
    """RM ANOVA test with pingouin.

    Eta-squared and omega-squared share the same suggested ranges for effect size classification:
    Low (0.01 – 0.059)
    Medium (0.06 – 0.139)
    Large (0.14+)
    Omega is considered a better measure of effect size than eta because it is unbiased in it’s calculation.

                        F    PR(>F)    df    eta_sq   mean_sq  omega_sq    sum_sq
        cnn       5.055341  0.000000  15.0  0.344949  0.000139  0.275461  0.002084
        text cnn  4.298028  0.000001  15.0  0.309255  0.000080  0.236169  0.001198
        dcnn      3.555700  0.000032  15.0  0.270278  0.000102  0.193286  0.001528

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
    """

    def _add_subjects(x):
        x['subject'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return x
    # Add 'subjects' to all models
    data = data.groupby(['model_name', exp_param]).apply(_add_subjects)
    if exp_param != 'model_name':
        # Get the list of models and ranges of experiment
        model_names = data['model_name'].unique()
        results_dict = dict()
        for model in model_names:
            # Create a list of all experiment values for this model
            model_values_list = data.loc[(data['model_name'] == model)]

            # Run ANOVA
            aov = pg.rm_anova(model_values_list, dv=metric, within=exp_param,
                              subject='subject', correction=True, detailed=True, effsize="np2")

            # Add this models results to the dict
            results_dict[model] = aov.iloc[0].to_dict()

            # Add power analysis
            effect, exp_n, power = anova_power_analysis(model_values_list, exp_param, metric, alpha=sig_level, power=0.8)
            power_analysis_cols = {'cohen_f': effect, 'n': len(model_values_list), 'exp_n': exp_n, 'power': power, 'exp_power': 0.8}
            results_dict[model] = {**results_dict[model], **power_analysis_cols}

        # Create dataframe
        anova_frame = pd.DataFrame.from_dict(results_dict, orient='columns').T
        if show_result:
            if all(p_value <= sig_level for p_value in anova_frame['p-GG-corr']):
                print("All models have significant p-values when comparing " + str(exp_param) + " groups.")
            else:
                print("The following models do not have significant p-values when comparing " + str(exp_param) + " groups.")
                print(anova_frame.loc[anova_frame['p-GG-corr'] > sig_level])

            return anova_frame
    else:
        # Run ANOVA
        aov = pg.rm_anova(data, dv=metric, within=exp_param,
                          subject='subject', correction=True, detailed=True, effsize="np2")

        # Add power analysis
        effect, exp_n, power = anova_power_analysis(data, exp_param, metric, alpha=sig_level, power=0.8)
        power_analysis_cols = {'cohen_f': effect, 'n': len(data), 'exp_n': exp_n, 'power': power, 'exp_power': 0.8}
        aov = aov.assign(**power_analysis_cols)

    if show_result:

        p_value = aov.iloc[0]['p-GG-corr']
        if p_value <= sig_level:
            print("The models have significant p-values when comparing groups. P-value = " + str(round(p_value, 5)))
        else:
            print("The models do not have significant p-values when comparing groups. P-value = " + str(round(p_value, 5)))

        return aov


def rm_two_way_anova_test(data, exp_param, metric, sig_level=0.05, show_result=True):
    """RM ANOVA test with pingouin.

    Eta-squared and omega-squared share the same suggested ranges for effect size classification:
    Low (0.01 – 0.059)
    Medium (0.06 – 0.139)
    Large (0.14+)
    Omega is considered a better measure of effect size than eta because it is unbiased in it’s calculation.

                        F    PR(>F)    df    eta_sq   mean_sq  omega_sq    sum_sq
        cnn       5.055341  0.000000  15.0  0.344949  0.000139  0.275461  0.002084
        text cnn  4.298028  0.000001  15.0  0.309255  0.000080  0.236169  0.001198
        dcnn      3.555700  0.000032  15.0  0.270278  0.000102  0.193286  0.001528

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (list): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        anova_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
    """

    # Add 'subjects' to all models
    data = data.groupby(['model_name'] + exp_param).apply(_add_subjects)

    # Get the list of models and ranges of experiment
    model_names = data['model_name'].unique()
    results_dict = []
    for model in model_names:
        # Create a list of all experiment values for this model
        model_values_list = data.loc[(data['model_name'] == model)]

        # Run ANOVA
        aov = pg.rm_anova(model_values_list, dv=metric, within=exp_param,
                          subject='subject', correction=True, detailed=True, effsize="np2")
        aov.insert(0, 'model_name', model)

        # Add this models results to the dict
        results_dict.append(aov)

    # Create dataframe
    anova_frame = pd.concat(results_dict, axis=0).reset_index(drop=True)

    if show_result:
        if all(p_value <= sig_level for p_value in anova_frame['p-GG-corr']):
            print("All models have significant p-values when comparing " + str(exp_param) + " groups.")
        else:
            print("The following models do not have significant p-values when comparing " + str(exp_param) + " groups.")
            print(anova_frame.loc[anova_frame['p-GG-corr'] > sig_level])

        return anova_frame


def tukey_hsd(data, exp_param, metric, sig_level=0.05, show_result=True):
    """ANOVA and Tukey HSD post-hoc comparison from statsmodels.

    The Tukey HSD post-hoc comparison test controls for type I error and maintains the family-wise error rate at 0.05.
    The group1 and group2 columns are the groups being compared,
    meandiff column is the difference in means of the two groups being calculated as group2 – group1,
    lower/upper columns are the lower/upper boundaries of the 95% confidence interval,
    reject column states whether or not the null hypothesis should be rejected.

            model_name  group1  group2  meandiff  p-value   lower   upper  reject
        0          cnn     500    1000    0.0014   0.9000 -0.0021  0.0050   False
        1          cnn     500    1500    0.0041   0.0083  0.0006  0.0077    True
        2          cnn     500    2000    0.0051   0.0010  0.0016  0.0087    True

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        tukey_frame (Dataframe): Contains f-statistic, p-value and eta/omega effect sizes.
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


def wilcoxon_rank(data, exp_param, metric, sig_level=0.05, show_result=True):
    """Scipy Wilcoxon signed ranks test for two experiment groups.

    Args:
        data (Dataframe): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        w_test_frame (Dataframe): Columns are t-statistic and p-value.
    """

    # Ensure only two groups are being tested
    if len(data[exp_param].unique()) != 2:
        raise ValueError("Too many groups in groups column! Found " + str(len(data[exp_param].unique())) + " but should == 2.")

    # Create results frame
    w_test_frame = pd.DataFrame(columns=['model_name', 't-stat', 'p-value'])

    # Get the list of models and experiment params
    model_names = data['model_name'].unique()
    groups = data[exp_param].unique()
    for model in model_names:

        # Select the data to compare
        data_a = data.loc[(data['model_name'] == model) & (data[exp_param] == groups[0])][metric]
        data_b = data.loc[(data['model_name'] == model) & (data[exp_param] == groups[1])][metric]

        # Perform power analysis
        effect, exp_n, act_power = t_test_power_analysis(data_a, data_b, alpha=sig_level, power=0.8)

        # T-test
        t, p = wilcoxon(data_a, y=data_b)

        # Append to result frame
        w_test_frame = w_test_frame.append({'model_name': model, 't-stat': t, 'p-value': p, 'cohen-d': effect,
                                            'n': len(data_a), 'exp_n': exp_n, 'power': act_power, 'exp_power': 0.8},
                                           ignore_index=True)

    if show_result:
        if all(p_value <= sig_level for p_value in w_test_frame['p-value']):
            print("All models have significant p-values when comparing " + exp_param + " groups.")
        else:
            print("The following models do not have significant p-values when comparing " + exp_param + " groups.")
            print(w_test_frame.loc[w_test_frame['p-value'] > sig_level])

    return w_test_frame


def mcnemar_test(pred_a, pred_b, exact=False, correction=True, show_result=True):
    """Mcnemar's test of homogeneity from statsmodels.

    Args:
        pred_a (Dataframe): Predictions of model A with columns 'true', and 'predicted'.
        pred_b (Dataframe): Predictions of model B with columns 'true', and 'predicted'.
        exact (bool): If exact is true, then the binomial distribution will be used. If exact is false,
            then the chisquare distribution will be used, which is the approximation to the distribution of the test
             statistic for large sample sizes.
        correction (bool): If true, then a continuity correction is used for the chisquare distribution (if exact is false.)
        show_result (bool): Whether to print the results of the test. Default=True.

    Returns:
        statistic (float): The test statistic is the chisquare statistic if exact is false.
        p-value (float): p-value of the null hypothesis of equal marginal distributions.
    """

    # Check predictions are the same length
    assert len(pred_a) == len(pred_b)

    # Create a contingency table
    table = pd.DataFrame([[0, 0], [0, 0]], index=['B_corr', 'B_wrong'], columns=['A_corr', 'A_wrong'])

    for i in range(len(pred_a)):
        # Get the true label
        true_lbl = pred_a['true'].iloc[i]

        # A and B correct
        if pred_a['predicted'].iloc[i] == true_lbl and pred_b['predicted'].iloc[i] == true_lbl:
            table.loc['B_corr', 'A_corr'] += 1
        # A and B incorrect
        elif pred_a['predicted'].iloc[i] != true_lbl and pred_b['predicted'].iloc[i] != true_lbl:
            table.loc['B_wrong', 'A_wrong'] += 1
        # A correct and B incorrect
        elif pred_a['predicted'].iloc[i] == true_lbl and pred_b['predicted'].iloc[i] != true_lbl:
            table.loc['B_wrong', 'A_corr'] += 1
        # A incorrect and B correct
        elif pred_a['predicted'].iloc[i] != true_lbl and pred_b['predicted'].iloc[i] == true_lbl:
            table.loc['B_corr', 'A_wrong'] += 1

    results = mcnemar(table.to_numpy(), exact=exact, correction=correction)

    if show_result:
        print("Statistic = " + str(results.statistic) + " p-value = " + str(results.pvalue))

    return results.statistic, results.pvalue


def mcnemar_pairwise_test(path, models, exact=False, correction=True, bonf_corr=True, sig_level=0.05, show_result=False):
    """Repeated pairwise Mcnemar's test of homogeneity from statsmodels. Also uses Bonferroni correction for multiple tests.

    Args:
        path (str): Path to folder containing the model predictions.
        models (list): List of all models to conduct pairwise tests. Must be in path/model/model_predictions.csv.
        exact (bool): If exact is true, then the binomial distribution will be used. If exact is false,
            then the chisquare distribution will be used, which is the approximation to the distribution of the test
             statistic for large sample sizes.
        correction (bool): If true, then a continuity correction is used for the chisquare distribution (if exact is false.)
        bonf_corr (bool): Whether to apply Bonferroni correction. Default=True.
        sig_level (float): The test significance level. Default=0.05.
        show_result (bool): Whether to print the results of the test. Default=False.

    Returns:
        results (Dataframe): Dataframe contains pairwise comparisons per-row,
            with columns ['Model_A', 'Model_B', 'statistic', 'p-value', 'reject', 'p-corrected', 'bonf_alpha']
    """
    # Get all pairwise combinations
    combinations = list(itertools.combinations(models, 2))

    results_list = []
    # For each pair compute mcnemar's
    for pair in combinations:

        # Get the data
        m_a_data = load_predictions(os.path.join(path, pair[0], pair[0] + '_predictions.csv'))
        m_b_data = load_predictions(os.path.join(path, pair[1], pair[1] + '_predictions.csv'))

        # Run mcnemar
        t, p = mcnemar_test(m_a_data, m_b_data, exact=exact, correction=correction, show_result=False)

        # Create dict for this pair
        results_list.append({'Model_A': pair[0], 'Model_B': pair[1], 'statistic': t, 'p-value': p})

    # Create dataframe to hold results
    results = pd.DataFrame(results_list)

    # Conduct Bonferroni correction
    if bonf_corr:
        reject, p_corr, _, bonf_alpha = multipletests(results['p-value'].to_numpy(), alpha=sig_level, method='bonferroni', returnsorted=False)
        results['reject'] = reject
        results['p-corrected'] = p_corr
        results['bonf_alpha'] = bonf_alpha

    if show_result:
        print(results)
    return results


def pair_data(a, b):
    data = []
    for i in range(len(a)):
        data.append([a[i], b[i]])
    return np.array(data)


def bayes_signrank(data_a, data_b, model_a, model_b, rope=0.01, show_result=False):
    """Beyesian Sign Rank test from:
    Benavoli, A., Corani, G., Demšar, J. and Zaffalon, M. (2017)
    Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.
    Journal of Machine Learning Research

    Args:
        data_a (list): List with all scores for model A.
        data_b (list): List with all scores for model B.
        model_a (str): Name of model A.
        model_b (str): Name of model B.
        rope (float): The region of practical equivalence. We consider two classifiers equivalent if the difference in their performance is smaller than rope.
        show_result (bool): Whether to print the results of the test. Default=False.

    Returns:
        results (Dataframe): Dataframe contains pairwise comparisons per-row,
            with columns ['Model_A', 'Model_B', 'P({a} > {b})', 'P({a} == {b})', 'P({b} > {a})', 'left', 'within', 'right]
    """

    # Pair data into 2d array
    curr_data = pair_data(data_a, data_b)

    # Apply Beyesian sign rank
    left, within, right = bt.signrank(curr_data, rope=rope, verbose=False)
    result = pd.DataFrame({'model_A': model_a, 'model_B': model_b, 'P({a} > {b})': left, 'P({a} == {b})': within,
                           'P({b} > {a})': right, 'left': left, 'within': within, 'right': right}, index=[0])
    result.reset_index(drop=True, inplace=True)
    if show_result:
        print('P({c1} > {c2}) = {pl}, P({c1} == {c2}) = {pe}, P({c2} > {c1}) = {pr}'.
              format(c1=model_a, c2=model_b, pl=left, pe=within, pr=right))

    return result


def pairwise_bayes_signrank(data, exp_param, metric, rope=0.01, show_result=False):
    """Repeated pairwise Beyesian Sign Rank test from:
    Benavoli, A., Corani, G., Demšar, J. and Zaffalon, M. (2017)
    Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.
    Journal of Machine Learning Research

    Args:
        data (DataFrame): Dataframe grouped by model_name and experiment_type values.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        rope (float): The region of practical equivalence. We consider two classifiers equivalent if the difference in their performance is smaller than rope.
        show_result (bool): Whether to print the results of the test. Default=False.

    Returns:
        results (Dataframe): Dataframe contains pairwise comparisons per-row,
            with columns ['Model_A', 'Model_B', 'P({a} > {b})', 'P({a} == {b})', 'P({b} > {a})', 'left', 'within', 'right]
    """
    # Get all pairwise combinations
    combinations = list(itertools.combinations(data[exp_param].unique(), 2))
    combinations.sort()

    results_list = []
    # For each pair compute mcnemar's
    for pair in combinations:

        # Get the data for this pair
        a = data.loc[(data[exp_param] == pair[0])][metric].tolist()
        b = data.loc[(data[exp_param] == pair[1])][metric].tolist()

        # Run Bayes for this pair
        results_list.append(bayes_signrank(a, b, pair[0], pair[1], rope=rope, show_result=False))

    results = pd.concat(results_list)
    results.reset_index(drop=True, inplace=True)
    if show_result:
        print(results)
    return results


def multi_pairwise_bayes_signrank(data, items, exp_param, metric, rope=0.01, show_result=False):
    """Multiple per-item repeated pairwise Beyesian Sign Rank test from:
    Benavoli, A., Corani, G., Demšar, J. and Zaffalon, M. (2017)
    Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.
    Journal of Machine Learning Research

    Args:
        data (DataFrame): Dataframe grouped by model_name and experiment_type values.
        items (string): Key in dataframe to repeatedly conduct pairwise tests, e.g. model_name.
        exp_param (string): Indicates which columns values to group data for comparison i.e. vocab_size.
        metric (string): Indicates which column name has the result values i.e. test_acc.
        rope (float): The region of practical equivalence. We consider two classifiers equivalent if the difference in their performance is smaller than rope.
        show_result (bool): Whether to print the results of the test. Default=False.

    Returns:
        results (Dataframe): Dataframe contains pairwise comparisons per-row,
            with columns ['Model_A', 'Model_B', 'P({a} > {b})', 'P({a} == {b})', 'P({b} > {a})', 'left', 'within', 'right]
    """
    # Get all pairwise combinations
    combinations = list(itertools.combinations(data[exp_param].unique(), 2))
    combinations.sort()

    # For each item compute pairwise mcnemar's
    results_list = []
    for item in data[items].unique():
        current_data = data.loc[(data[items] == item)]

        item_results_list = []
        # For each pair compute mcnemar's
        for pair in combinations:

            # Get the data for this pair
            a = current_data.loc[(current_data[exp_param] == pair[0])][metric].tolist()
            b = current_data.loc[(current_data[exp_param] == pair[1])][metric].tolist()

            # Run Bayes for this pair
            item_results_list.append(bayes_signrank(a, b, pair[0], pair[1], rope=rope, show_result=False))

        item_results = pd.concat(item_results_list)
        item_results.reset_index(drop=True, inplace=True)
        item_results.insert(0, items, item)

        results_list.append(item_results)

    results = pd.concat(results_list)
    results.reset_index(drop=True, inplace=True)
    if show_result:
        print(results)
    return results
