from data_processing import *
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

# Set the task and experiment type
task_name = 'swda'

# Set data dir
data_dir = os.path.join('..', task_name)


"""Process all input sequence data"""
params = ['vocab_size', 'max_seq_length']
if task_name == 'swda':
    params += ['use_punct', 'to_lower']

for exp_param in params:

    # Load experiment data
    data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))

    # Remove the numbered experiment names and replace '_' char
    data = data.drop('experiment_name', axis='columns')
    data.model_name = data.model_name.str.replace("_", " ")

    # Drop bi-directional models
    data = data[~data['model_name'].isin(['bi lstm', 'bi gru'])]

    # Sort by model name and experiment type
    sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'gru']
    data = sort_dataframe_by_list_and_param(data, 'model_name', sort_order, exp_param)

    # Multiply accuracy columns by 100
    acc_cols = data.filter(like='acc').columns
    data[acc_cols] *= 100

    # Save dataframe with all the data in
    save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

    # Group by model name and get means
    data_means = data.groupby(['model_name', exp_param], sort=True).mean()

    # Add std columns
    if task_name == 'maptask' or exp_param == 'to_lower':
        cols = params.copy()
        cols.append('use_punct')
        cols.remove(exp_param)
        data.drop(columns=cols, inplace=True)  # Need to drop columns to calc std
    data_std = data.groupby(['model_name', exp_param], sort=True).std()
    data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
    data_std = data_std.add_suffix('_std')
    data_means = data_means.merge(data_std, left_on=['model_name', exp_param], right_index=True)

    data_means.reset_index(inplace=True)
    data_means.model_name = data_means.model_name.str.replace("_", " ")
    data_means = sort_dataframe_by_list(data_means, 'model_name', sort_order)

    # Save dataframe with mean data in
    save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)


"""Process input sequence search data"""
exp_param = 'input_seq'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))

# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Drop bi-directional models
data = data[~data['model_name'].isin(['bi lstm', 'bi gru'])]

# Sort by model name
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'gru']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Multiply accuracy columns by 100
acc_cols = data.filter(like='acc').columns
data[acc_cols] *= 100

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

# Group by model name and get means
data_means = data.groupby(['model_name', 'vocab_size', 'max_seq_length'], sort=True).mean()
# Add std columns
data.drop(columns=['use_punct'], inplace=True)  # Need to drop columns to calc std
data_std = data.groupby(['model_name', 'vocab_size', 'max_seq_length'], sort=True).std()
data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
data_std = data_std.add_suffix('_std')
data_means = data_means.merge(data_std, left_on=['model_name', 'vocab_size', 'max_seq_length'], right_index=True)

data_means.reset_index(inplace=True)
data_means.model_name = data_means.model_name.str.replace("_", " ")
data_means = sort_dataframe_by_list(data_means, 'model_name', sort_order)

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)


"""Process embedding type data"""
exp_param = 'embedding_type'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))

# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Drop bi-directional models
data = data[~data['model_name'].isin(['bi lstm', 'bi gru'])]

# Sort by model name
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'gru']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Multiply accuracy columns by 100
acc_cols = data.filter(like='acc').columns
data[acc_cols] *= 100

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

# Group by model name and get means
data_means = data.groupby(['model_name', exp_param, 'embedding_dim'], sort=True).mean()
# Add std columns
data.drop(columns=['vocab_size', 'max_seq_length', 'use_punct'], inplace=True)  # Need to drop columns to calc std
data_std = data.groupby(['model_name', exp_param, 'embedding_dim'], sort=True).std()
data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
data_std = data_std.add_suffix('_std')
data_means = data_means.merge(data_std, left_on=['model_name', exp_param, 'embedding_dim'], right_index=True)

data_means.reset_index(inplace=True)
data_means.model_name = data_means.model_name.str.replace("_", " ")
data_means = sort_dataframe_by_list(data_means, 'model_name', sort_order)

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)

"""Process input sequence final data"""
exp_param = 'input_seq_final'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))

# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name
sort_order = ['cnn', 'text cnn', 'dcnn', 'rcnn', 'lstm', 'gru']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Multiply accuracy columns by 100
acc_cols = data.filter(like='acc').columns
data[acc_cols] *= 100

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

# Group by model name and get means
data_means = data.groupby(['model_name'], sort=True).mean()

# Add std columns
data.drop(columns=['vocab_size', 'max_seq_length', 'use_punct'], inplace=True)  # Need to drop columns to calc std
data_std = data.groupby(['model_name'], sort=True).std()
data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
data_std = data_std.add_suffix('_std')
data_means = data_means.merge(data_std, left_on=['model_name'], right_index=True)

data_means.reset_index(inplace=True)
data_means.model_name = data_means.model_name.str.replace("_", " ")
data_means = sort_dataframe_by_list(data_means, 'model_name', sort_order)

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)

"""Process model variants data"""
exp_param = 'model_variants'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_' + exp_param + '.csv'))


# Add layer number to model name rather than experiment name
def add_lyr_num(x):
    try:
        if '2lyr' in x.experiment_name:
            x.model_name += '_2lyr'
        elif '3lyr' in x.experiment_name:
            x.model_name += '_3lyr'
        else:
            x.model_name = x.model_name
        return x.model_name
    except KeyError:
        return x.experiment_name


data.model_name = data.apply(add_lyr_num, axis=1)

# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name
sort_order = ['cnn attn', 'text cnn attn', 'dcnn attn', 'rcnn attn', 'lstm attn', 'lstm' 'gru attn']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Multiply accuracy columns by 100
acc_cols = data.filter(like='acc').columns
data[acc_cols] *= 100

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_data.csv'), data)

# Group by model name and get means
data_means = data.groupby(['model_name'], sort=True).mean()

# Add std columns
data.drop(columns=['vocab_size', 'max_seq_length', 'use_punct'], inplace=True)  # Need to drop columns to calc std
data_std = data.groupby(['model_name'], sort=True).std()
data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
data_std = data_std.add_suffix('_std')
data_means = data_means.merge(data_std, left_on=['model_name'], right_index=True)

data_means.reset_index(inplace=True)
data_means.model_name = data_means.model_name.str.replace("_", " ")

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, exp_param, exp_param + '_mean_data.csv'), data_means)

"""Process language model data"""
exp_param = 'embedding_type'
# Load experiment data
data = load_dataframe(os.path.join(data_dir, task_name + '_language_models.csv'))
# Remove the numbered experiment names and replace '_' char
data = data.drop('experiment_name', axis='columns')
data.model_name = data.model_name.str.replace("_", " ")

# Sort by model name and experiment type
sort_order = ['bert', 'roberta', 'gpt2', 'dialogpt', 'xlnet', 'convert', 'elmo', 'use', 'mlstm char lm', 'nnlm']
data = sort_dataframe_by_list(data, 'model_name', sort_order)

# Multiply accuracy columns by 100
acc_cols = data.filter(like='acc').columns
data[acc_cols] *= 100

# Save dataframe with all the data in
save_dataframe(os.path.join(task_name, 'language_models', 'language_models_data.csv'), data)

# Get means over all experiments
data_means = data.groupby(exp_param, sort=False).mean()
# Add std columns
data.drop(columns=['vocab_size', 'max_seq_length', 'use_punct'], inplace=True)  # Need to drop columns to calc std
data_std = data.groupby(exp_param, sort=False).std()
data_std = data_std.drop(data_std.columns.difference(data_std.columns[data_std.columns.get_loc('train_loss'):]), axis=1)
data_std = data_std.add_suffix('_std')
data_means = data_means.merge(data_std, left_on=[exp_param], right_index=True)

data_means.reset_index(inplace=True)
data_means.insert(0, 'model_name', data_means['embedding_type'])
data_means.model_name = data_means.model_name.str.replace("_", " ")

# Save dataframe with mean data in
save_dataframe(os.path.join(task_name, 'language_models', 'language_models_mean_data.csv'), data_means)
