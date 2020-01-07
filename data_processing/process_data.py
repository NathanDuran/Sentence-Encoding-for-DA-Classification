from data_processing import *

pd.options.display.width = 0

task_name = 'swda'
experiment_type = 'vocab_size'

vocab_size_data = load_experiment_data(task_name, experiment_type)

vocab_size_means = get_experiment_means(vocab_size_data, experiment_type)

print(vocab_size_means)
print(vocab_size_means['test_acc'].max())

plot_line_chart(vocab_size_data, x="vocab_size", y="val_acc", hue="model_name", y_label='Test Accuracy', x_label='Vocabulary Size')