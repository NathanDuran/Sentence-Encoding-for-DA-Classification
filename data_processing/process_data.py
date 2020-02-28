import os
import pandas as pd
from data_processing import *

pd.options.display.width = 0

# Set the task and experiment type
task_name = 'swda'
experiment_type = 'max_seq_length'
experiment_name = 'Sequence Length'

# Set data dir
data_dir = os.path.join('..', task_name)
# Set and create the output directory if id doesn't exist
output_dir = os.path.join(task_name, experiment_type)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + experiment_type + '.csv'))
# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name and experiment type # TODO do i really need to sort future experiments?
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'bi lstm', 'gru', 'bi gru']
data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, experiment_type)
# Save dataframe with all the data in
save_dataframe(os.path.join(output_dir, experiment_type + '_data_raw.csv'), data)

# Get means over all experiments
data_means = get_means(data, experiment_type)
save_dataframe(os.path.join(output_dir, experiment_type + '_data_means.csv'), data_means)

# Get test and validation accuracy for each model
acc_data = data.drop(data.columns.difference(['model_name', experiment_type, 'val_acc', 'test_acc']), axis=1)
acc_data = acc_data.rename(columns={'val_acc': 'Val Acc', 'test_acc': 'Test Acc'})
acc_data = acc_data.melt(id_vars=['model_name', experiment_type])

g, fig = plot_lmplot_chart(acc_data, x=experiment_type, y="value", hue="model_name", col='variable',
                           order=5, num_legend_col=4, y_label='Accuracy', x_label=experiment_name,
                           share_x=True, num_col=1, colour='Paired')
fig.show()
g.savefig(os.path.join(output_dir, experiment_type + '_accuracy.png'))

# # Get max val_acc and test_acc/F1 for experiment_type per model into table
print("========================= Raw Data =========================")
max_of_raw_data = get_max(data, experiment_type) # TODO Don't bother with raw data?
print(max_of_raw_data)
print("Best validation accuracy in raw data:")
print(max_of_raw_data.loc[[max_of_raw_data['val_acc'].idxmax()], ['model_name', 'val_' + experiment_type, 'val_acc']])
print("Best test accuracy in raw data:")
print(max_of_raw_data.loc[[max_of_raw_data['test_acc'].idxmax()], ['model_name', 'test_' + experiment_type, 'test_acc', 'f1_micro', 'f1_weighted']])
save_dataframe(os.path.join(output_dir, experiment_type + '_max_of_raw_data.csv'), max_of_raw_data)
fig = plot_table(max_of_raw_data, title=experiment_name + ' Raw Data')
fig.show()

print("========================= Mean Data =========================")
max_of_mean_data = get_max(data_means, experiment_type)
print(max_of_mean_data)
print("Best validation accuracy in mean data:")
print(max_of_mean_data.loc[[max_of_mean_data['val_acc'].idxmax()], ['model_name', 'val_' + experiment_type, 'val_acc']])
print("Best test accuracy in mean data:")
print(max_of_mean_data.loc[[max_of_mean_data['test_acc'].idxmax()], ['model_name', 'test_' + experiment_type, 'test_acc', 'f1_micro', 'f1_weighted']])
save_dataframe(os.path.join(output_dir, experiment_type + '_max_of_mean_data.csv'), max_of_mean_data)
fig = plot_table(max_of_mean_data, title=experiment_name + ' Mean Data')
fig.show()

# Pairwise t-test between experiment parameters
# Bonferroni correction post-hoc comparison
# p-value/# of comparisons = 0.05/15 = 0.00333
# t_test_frame = pairwise_t_test(data, experiment_type, 'test_acc')
# print(t_test_frame)
#
# # One way ANOVA
# f_one_way_frame = f_oneway_test(data, experiment_type, 'test_acc')
# print(f_one_way_frame)
#
# # Pairwise ANOVA
# anova_test_frame = anova_test(data, experiment_type, 'test_acc')
# print(anova_test_frame)

# Tukeys HSD post-hoc comparison
for metric in ['val_acc', 'test_acc']:
    # Calculate the anova
    tukey_frame = tukey_hsd(data, experiment_type, metric)
    save_dataframe(os.path.join(output_dir, experiment_type + '_' + metric + '_anova.csv'), tukey_frame)

    # Drop the un-needed columns and generate heatmaps
    title = experiment_name + ' Validation Accuracy' if metric == 'val_acc' else experiment_name + ' Test Accuracy'
    tukey_frame = tukey_frame.drop(columns=['meandiff', 'lower', 'upper', 'reject'], axis=1)
    g, fig = plot_facetgrid(tukey_frame, x='group1', y='group2', hue='p-value', col='model_name', kind='heatmap',
                            title=title, y_label='', x_label='', num_col=2, colour='RdBu_r',
                            annot=True, fmt='0.3', linewidths=0.5, cbar=False, custom_boundaries=[0.0, 0.05, 1.0],
                            y_tick_rotation=45)
    fig.show()
    g.savefig(os.path.join(output_dir, experiment_type + '_' + metric + '_anova.png'))