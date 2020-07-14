from data_processing import *

# Set the task and experiment type
task_name = 'swda'

# Set data dir
data_dir = os.path.join('..', task_name)

# Process all input sequence data
for exp_param in ['vocab_size', 'max_seq_length', 'use_punct']:
    # Load experiment data
    data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))
    # Remove the numbered experiment names and replace '_' char
    data = data.drop('experiment_name', axis='columns')
    data.model_name = data.model_name.str.replace("_", " ")

    # Sort by model name and experiment type
    sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'bi lstm', 'gru', 'bi gru']
    data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, exp_param)

    # Save dataframe with all the data in
    save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

    # Group by model name and get means
    data_means = data.groupby(['model_name', exp_param], sort=True).mean()
    data_means.reset_index(inplace=True)
    data_means.model_name = data_means.model_name.str.replace("_", " ")

    # Save dataframe with mean data in
    save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)


# Process input sequence search data
exp_param = 'input_seq'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))
# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'gru']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

# Group by model name and get means
data_means = data.groupby(['model_name', 'vocab_size', 'max_seq_length', 'use_punct'], sort=True).mean()
data_means.reset_index(inplace=True)
data_means.model_name = data_means.model_name.str.replace("_", " ")

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)

# TODO Embeddings

# Process language model data
exp_param = 'embedding_type'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_language_models.csv'))
# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name and experiment type
sort_order = ['elmo', 'bert', 'use', 'nnlm', 'mlstm_char_lm'] # TODO change this order ['bert', 'bert_large', 'elmo', 'convert', 'use', 'mlstm_char_lm', 'nnlm']
data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, exp_param)

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, 'language_models', 'language_models_data.csv'), data)

# Get means over all experiments
data_means = data.groupby(exp_param, sort=False).mean()
data_means.reset_index(inplace=True)
data_means.insert(0, 'model_name', data_means['embedding_type'])
data_means.model_name = data_means.model_name.str.replace("_", " ")

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, 'language_models', 'language_models_mean_data.csv'), data_means)
