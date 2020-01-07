import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import six


def plot_line_chart(data, x, y, hue, title='', y_label='', x_label='', legend_title='Model'):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # Create line chart
    g = sns.lineplot(x=x, y=y, hue=hue, data=data, markers=True, ci=16)

    # Replace the '_' char in the data labels
    legend_labels = data[hue].unique().tolist()
    legend_labels = [label.replace("_", " ") for label in legend_labels]
    # Set legend position and labels
    handles, labels = g.get_legend_handles_labels()
    g.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, handles=handles[1:],
             labels=legend_labels, title=legend_title)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return plt


def plot_grouped_bar_chart(data, title='', y_label='', x_label=''):
    # Need to reshape dataframe from 'wide' to 'long' format
    data = data.copy()
    data.columns = data.columns.droplevel()  # Drops the multi-level column name
    data = data.reset_index().melt(id_vars=["index"]).sort_values(['variable', 'value'])
    data = data.sort_index()

    sns.set(style="whitegrid")
    g = sns.catplot(data=data, x='index', y='value', hue='variable',
                    kind="bar", palette="Paired", height=6, aspect=2, legend=False)
    g.despine(left=True)
    g.ax.legend(frameon=False, loc='upper right', ncol=1)
    g.set_xticklabels(rotation=45)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()
    return plt


def plot_table(data, title='AP Distances'):
    # Matplotlib refuses to show the row labels so just set them as a column
    data = data.copy()
    data.insert(0, column='Labels', value=data.index)

    col_width = 2.1
    row_height = 0.5
    font_size = 14
    header_color = '#40466e'
    row_colors = ['#f1f1f2', 'w']
    edge_color = 'w'
    bbox = [0, 0, 1, 1]
    header_columns = 1

    size = (np.array(data.shape[::]) + np.array([0, 1])) * np.array([col_width, row_height])
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')

    table = ax.table(cellText=data.values, cellLoc='center', colLabels=data.columns, bbox=bbox)

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for k, cell in six.iteritems(table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    plt.title(title, fontdict={'fontsize': font_size * 2})

    return fig
