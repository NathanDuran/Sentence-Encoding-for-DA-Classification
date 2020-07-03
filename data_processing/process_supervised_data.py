import os
import pandas as pd
from data_processing import *

pd.options.display.width = 0

# Set the task and experiment type
task_name = 'swda'
experiment_type = 'vocab_size'
experiment_name = 'Vocabulary Size'

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
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'bi lstm', 'gru', 'bi gru']
exp_param = 'embedding_dim' if 'embeddings' in experiment_type else experiment_type
data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, exp_param)

# Save dataframe with all the data in
save_dataframe(os.path.join(output_dir, experiment_type + '_data_raw.csv'), data)

# Get means over all experiments
data_means = get_means(data, exp_param)
save_dataframe(os.path.join(output_dir, experiment_type + '_data_means.csv'), data_means)

# Get test and validation accuracy for each model
acc_data = data.drop(data.columns.difference(['model_name', exp_param, 'val_acc', 'test_acc']), axis=1)
acc_data = acc_data.rename(columns={'val_acc': 'Val Acc', 'test_acc': 'Test Acc'})
acc_data = acc_data.melt(id_vars=['model_name', exp_param])

if exp_param == 'use_punct':
    g, fig = plot_facetgrid(acc_data, x=exp_param, y="value", hue="model_name", col='variable', kind='violin',
                            num_legend_col=4, y_label='Accuracy', x_label=experiment_name,
                            share_y=True, num_col=1, colour='Paired')
else:
    g, fig = plot_relplot(acc_data, x=exp_param, y='value', hue='model_name', col='variable', kind='line', ci=95,
                          title='', y_label='Accuracy', x_label='Vocabulary Size',  share_x=True, share_y=False, num_col=1,
                          legend_loc='lower right', num_legend_col=4, colour='Paired')
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
fig = plot_table(max_of_mean_data, title=experiment_name + ' Mean Data')
fig.show()

for metric in ['val_acc', 'test_acc']:
    # Set the title for graphs and dataframes
    title = experiment_name + ' Validation Accuracy' if metric == 'val_acc' else experiment_name + ' Test Accuracy'

    if experiment_type != 'use_punct':
        # Tukeys HSD post-hoc comparison
        tukey_frame = tukey_hsd(data, exp_param, metric)
        save_dataframe(os.path.join(output_dir, experiment_type + '_' + metric + '_anova.csv'), tukey_frame)

        # Drop the un-needed columns and generate heatmaps
        tukey_frame = tukey_frame.drop(columns=['meandiff', 'lower', 'upper', 'reject'], axis=1)
        # TODO Remove vocab_size > 5000 to make plots nicer
        if exp_param == 'vocab_size':
            tukey_frame.drop(tukey_frame[(tukey_frame.group1 > 5000) | (tukey_frame.group2 > 5000)].index, inplace=True)
        g, fig = plot_facetgrid(tukey_frame, x='group1', y='group2', hue='p-value', col='model_name', kind='heatmap',
                                title=title, y_label='', x_label='', num_col=2, colour='RdBu_r',
                                annot=True, fmt='0.3', linewidths=0.5, cbar=False, custom_boundaries=[0.0, 0.05, 1.0],
                                y_tick_rotation=45, height=4)
        fig.show()
        g.savefig(os.path.join(output_dir, experiment_type + '_' + metric + '_anova.png'))
    else:
        # One way t-test
        f_one_way_frame = f_oneway_test(data, exp_param, 'test_acc')
        f_one_way_frame.columns = pd.MultiIndex.from_product([[title], f_one_way_frame.columns])
        print(f_one_way_frame)
        save_dataframe(os.path.join(output_dir, experiment_type + '_' + metric + '_t-test.csv'), f_one_way_frame)