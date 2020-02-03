from data_processing import *

pd.options.display.width = 0

task_name = 'swda'
experiment_type = 'vocab_size'

# Load experiment data
vocab_size_data = load_experiment_data(task_name, experiment_type)
# Sort by model name and experiment type
vocab_size_data = sort_experiment_data_by_model_and_metric(vocab_size_data, experiment_type)
# Save dataframe
save_dataframe(experiment_type + '_current.csv', vocab_size_data)
# Generate and save charts # TODO Add saving and plot test and val on same plot?
# g, fig = plot_implot_chart(vocab_size_data, x="vocab_size", y="val_acc", hue="model_name",
#                            order=4, num_legend_col=4, y_label='Test Accuracy', x_label='Vocabulary Size',
#                            legend_loc='lower right', colour='Paired')
# fig.show()
# g, fig = plot_implot_chart(vocab_size_data, x="vocab_size", y="test_acc", hue="model_name",
#                            order=4, num_legend_col=4, y_label='Test Accuracy', x_label='Vocabulary Size',
#                            legend_loc='lower right', colour='Paired')
# fig.show()

# Get means over all experiments
vocab_size_means = get_experiment_means(vocab_size_data, experiment_type)
# print(vocab_size_means)
print(vocab_size_means['test_acc'].max())
print(vocab_size_means['val_acc'].max())

# Get best (val acc and test acc/F1) for experiment_type per model into table
best_val = vocab_size_means.loc[vocab_size_means.groupby(['model_name'], sort=False)['val_acc'].idxmax()]
best_val.drop(best_val.columns.difference(['model_name', 'vocab_size', 'val_acc']), 1, inplace=True)
best_val.rename(columns={'vocab_size': 'val_vocab_size'}, inplace=True)
best_val = best_val.set_index('model_name')

best_test = vocab_size_means.loc[vocab_size_means.groupby(['model_name'], sort=False)['test_acc'].idxmax()]
best_test.drop(best_test.columns.difference(['model_name', 'vocab_size', 'test_acc', 'f1_micro', 'f1_weighted']), 1, inplace=True)
best_test.rename(columns={'vocab_size': 'test_vocab_size'}, inplace=True)
best_test = best_test.set_index('model_name')

vocab_size_best = pd.concat([best_val, best_test], axis=1, ignore_index=False, sort=False).reset_index()
print(vocab_size_best)

# Test if result pairs are statistically significant # TODO concat val and test data? + graph to show what is significant
t_test_frame = t_test(vocab_size_data, experiment_type)
print(t_test_frame)
# TODO Change order so rcnn is after dcnn (also for models/model_params)