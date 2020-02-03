from data_processing import *

pd.options.display.width = 0

task_name = 'swda'
experiment_type = 'vocab_size'

# TODO Save vocab size .csv ordered by vocab size and model (because I did them out of order)
#  (also put rcnn after dcnn - here, when running experiments and also in models)
vocab_size_data = load_experiment_data(task_name, experiment_type)
print(vocab_size_data['test_acc'].max())
vocab_size_means = get_experiment_means(vocab_size_data, experiment_type)

print(vocab_size_means)
print(vocab_size_means['test_acc'].max())

# Get best vocab size (val/test) per model into table TODO (add min and max for each)
best_val = vocab_size_means.loc[vocab_size_means.groupby(['model_name'], sort=False)['val_acc'].idxmax()]
best_val.drop(best_val.columns.difference(['model_name', 'vocab_size', 'val_acc']), 1, inplace=True)
best_val.rename(columns={'vocab_size': 'val_vocab_size'}, inplace=True)
best_val = best_val.set_index('model_name')

best_test = vocab_size_means.loc[vocab_size_means.groupby(['model_name'], sort=False)['test_acc'].idxmax()]
best_test.drop(best_test.columns.difference(['model_name', 'vocab_size', 'test_acc']), 1, inplace=True)
best_test.rename(columns={'vocab_size': 'test_vocab_size'}, inplace=True)
best_test = best_test.set_index('model_name')

vocab_size_best = pd.concat([best_val, best_test], axis=1, ignore_index=False, sort=False).reset_index()
print(vocab_size_best)

g, fig = plot_implot_chart(vocab_size_data, x="vocab_size", y="val_acc", hue="model_name",
                           order=4, num_legend_col=4, y_label='Test Accuracy', x_label='Vocabulary Size',
                           legend_loc='lower right', colour='Paired')
fig.show()
