import os
import pandas as pd
from data_processing import *

pd.options.display.width = 0

# Set the task and experiment type
task_name = 'swda'
experiment_type = 'language_models'
experiment_name = 'Language Models'

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

# Sort by model name and experiment type
sort_order = ['elmo', 'bert', 'use', 'nnlm', 'mlstm_char_lm']
exp_param = 'embedding_type'
data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, exp_param)

# Save dataframe with all the data in
save_dataframe(os.path.join(output_dir, experiment_type + '_data_raw.csv'), data)

# Get means over all experiments
data_means = data.groupby(exp_param, sort=False).mean()
data_means.reset_index(inplace=True)
data_means.insert(0, 'model_name', data_means['embedding_type'])
data_means.model_name = data_means.model_name.str.replace("_", " ")
save_dataframe(os.path.join(output_dir, experiment_type + '_data_means.csv'), data_means)

# Get test and validation accuracy for each model
acc_data = data.drop(data.columns.difference(['model_name', exp_param, 'val_acc', 'test_acc']), axis=1)
acc_data = acc_data.rename(columns={'val_acc': 'Val Acc', 'test_acc': 'Test Acc'})
acc_data = acc_data.melt(id_vars=['model_name', exp_param])

g, fig = plot_facetgrid(acc_data, x="model_name", y="value", hue="model_name", col='variable', kind='violin',
                        num_legend_col=5, y_label='Accuracy', x_label=experiment_name,
                        share_y=False, num_col=1, colour='Paired', dodge=False)

fig.show()
g.savefig(os.path.join(output_dir, experiment_type + '_accuracy.png'))

# # Get max val_acc and test_acc/F1 for experiment_type per model into table
print("========================= Raw Data =========================")
max_of_raw_data = get_max(data, exp_param)  # TODO Don't bother with raw data?
print(max_of_raw_data)
print("Best validation accuracy in raw data:")
print(max_of_raw_data.loc[[max_of_raw_data['val_acc'].idxmax()], ['model_name', 'val_' + exp_param, 'val_acc']])
print("Best test accuracy in raw data:")
print(max_of_raw_data.loc[[max_of_raw_data['test_acc'].idxmax()], ['model_name', 'test_' + exp_param, 'test_acc', 'f1_micro', 'f1_weighted']])
save_dataframe(os.path.join(output_dir, experiment_type + '_max_of_raw_data.csv'), max_of_raw_data)
max_of_raw_data.drop(['val_embedding_type', 'test_embedding_type'], axis='columns', inplace=True)
fig = plot_table(max_of_raw_data, title=experiment_name + ' Raw Data')
fig.show()

print("========================= Mean Data =========================")
max_of_mean_data = get_max(data_means, exp_param)
print(max_of_mean_data)
print("Best validation accuracy in mean data:")
print(max_of_mean_data.loc[[max_of_mean_data['val_acc'].idxmax()], ['model_name', 'val_' + exp_param, 'val_acc']])
print("Best test accuracy in mean data:")
print(max_of_mean_data.loc[[max_of_mean_data['test_acc'].idxmax()], ['model_name', 'test_' + exp_param, 'test_acc', 'f1_micro', 'f1_weighted']])
save_dataframe(os.path.join(output_dir, experiment_type + '_max_of_mean_data.csv'), max_of_mean_data)
max_of_mean_data.drop(['val_embedding_type', 'test_embedding_type'], axis='columns', inplace=True)
fig = plot_table(max_of_mean_data, title=experiment_name + ' Mean Data')
fig.show()