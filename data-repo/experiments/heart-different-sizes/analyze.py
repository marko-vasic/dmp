import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# UT Color pallete
ORANGE = '#bf5700'
LIGHT_ORANGE = '#f8971f'
GRAY = '#333f48'
LIGHT_GRAY = '#9cadb7'
YELLOW = '#ffd600'
LIGHT_GREEN = '#a6cd57'
GREEN = '#579d42'
LIGHT_BLUE = '#00a9b7'
BLUE = '#005f86'
CREAM = '#d6d2c4'


COLOR_PALETTE = [
    ORANGE,
    LIGHT_ORANGE,
    GRAY,
    LIGHT_GRAY,
    YELLOW,
    LIGHT_GREEN,
    GREEN,
    CREAM,
    LIGHT_BLUE,
    BLUE
]

# ERROR BAR 
MARKERSIZE=10
ELINEWIDTH=5
CAPSIZE=7
CAPTHICK=5


def load_data_averaged_across_networks(layers, versions):
    """
    Loads data where results for same number of layers is avereged out.
    """
    maxes = []
    means = []
    stds = []
    mins = []
    
    for layer in layers:
        data = None
        for version in versions:
            file_path = 'layers_{}{}/HEART_convergence_times.csv'.format(layer, version)
            new_data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
            if data is None:
                data = new_data
            else:
                data = np.concatenate((data, new_data))
    
        print('correlation (conf <> conv)            in {}: {:.2f}'.format(
            layer + version, pearsonr(data[:,0], data[:,1])[0]))
        print('correlation (abs(0.5 - conf) <> conv) in {}: {:.2f}'.format(
            layer + version, pearsonr(abs(0.5 - data[:,0]), data[:,1])[0]))
        
        assert data.shape[0] == 255 * len(versions)
        means.append(data[:,1].mean())
        stds.append(data[:,1].std())
        maxes.append(data[:,1].max())
        mins.append(data[:,1].min())

    return means, stds, maxes, mins


def load_data_convergence(column):
    layers = ['1', '2', '3', '4', '5']
    versions = ['a', 'b']
    configs = []
    all_data = []
    for layer in layers:
        for version in versions:
            config_name = layer + version
            file_path = 'layers_{}{}/HEART_convergence_times.csv'.format(layer, version)
            df = pd.read_csv(file_path, usecols=[column])
            configs.append(config_name)
            all_data.append(np.squeeze(df.values))
    return configs, all_data


def load_data_90production_per_layer():
    layers = ['5']
    versions = ['a', 'b']
    configs = []
    all_data = []
    for layer in layers:
        for version in versions:
            config_name = layer + version
            file_path = 'layers_{}{}/90production_times_per_layer.csv'.format(layer, version)
            df = pd.read_csv(file_path)
            configs.append(config_name)
            all_data.append(np.squeeze(df.values))
    return configs, all_data


def plot_mean_std(mu, sigma, fontsize=24):
    fig, ax = plt.subplots()
    xticks = np.arange(len(mu)) + 1
    ax.plot(xticks,
            mu,
            lw=4,
            # label='hello',
            color=GREEN)
    ax.fill_between(xticks,
                    mu + sigma,
                    mu - sigma,
                    facecolor=GREEN,
                    alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'\textbf{hidden layers}',
                  fontsize=fontsize,
                  color=GRAY)
    ax.set_ylabel(r'\textbf{convergence time}',
                  fontsize=fontsize,
                  color=GRAY)
    plt.xticks(xticks)
    ax.grid()

    plt.xticks(fontsize=fontsize, color=ORANGE, weight='bold')
    plt.yticks(fontsize=fontsize, color=ORANGE, weight='bold')
    
    plt.savefig('temp.pdf')
    plt.show()
    plt.close()


def box_plot(configs, data):
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data,
        showfliers=True,  # don't show outliers
        whis=(10,90),  # percentiles used for lower and upper whisker
        # conf_intervals=[[0,100]] * len(data),
    )
    # ax.xaxis.set_ticks(configs)
    ax.set_xticklabels(configs)
    plt.savefig('temp2.pdf')
    # plt.show()
    plt.close()


def error_bar(data,
              sort_by_mean=False,
              lower_percentile=10,
              upper_percentile=90,
              fontsize=24,
              out_file=None,
              ylabel=None):

    if sort_by_mean:
        means = np.mean(data, axis=1)
        for i in range(0, len(data), 2):
            if means[i] > means[i + 1]:
                temp = np.copy(data[i])
                data[i] = data[i + 1]
                data[i + 1] = temp
            
    fig, ax = plt.subplots()

    lower_error_bar = (np.mean(data, axis=1)
                       - np.percentile(data, lower_percentile, axis=1))
    lower_error_bar = np.reshape(lower_error_bar, (len(lower_error_bar), 1))

    higher_error_bar = (np.percentile(data, upper_percentile, axis=1)
                        - np.mean(data, axis=1))
    higher_error_bar = np.reshape(higher_error_bar, (len(higher_error_bar), 1))

    error_bars = np.concatenate((lower_error_bar, higher_error_bar), axis=1).T

    x_ticks = np.expand_dims((np.arange(len(data) / 2) + 1), axis=1)
    x_ticks = np.concatenate((x_ticks - 0.1, x_ticks + 0.1), axis=1).reshape((len(data),))

    for i in range(len(x_ticks)):
        ax.errorbar(x_ticks[i],
                    np.mean(data, axis=1)[i],
                    yerr=np.reshape(error_bars[:,i], (2,1)),
                    ecolor=COLOR_PALETTE[i],
                    fmt='o',
                    markersize=MARKERSIZE,
                    markerfacecolor='w',
                    markeredgecolor=GRAY,
                    elinewidth=ELINEWIDTH,
                    capsize=CAPSIZE,
                    capthick=CAPTHICK
        )

    ax.set_xlabel(r'\textbf{hidden layers}',
                  fontsize=fontsize,
                  color=GRAY)
    if ylabel:
        ax.set_ylabel('\\textbf{{{}}}'.format(ylabel),
                      fontsize=fontsize,
                      color=GRAY)
    ax.grid()

    plt.xticks(fontsize=fontsize, color=ORANGE, weight='bold')
    plt.yticks(fontsize=fontsize, color=ORANGE, weight='bold')

    plt.tight_layout()
    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    

def error_bar_90production_per_layer(data,
                                     fontsize=24):
    data = np.swapaxes(data,0,1).reshape((12,3))
    mean = data[:,1]
    percentile90 = data[:,0]
    percentile10 = data[:,2]

    lower_error_bar = mean - percentile10
    lower_error_bar = lower_error_bar.reshape((lower_error_bar.shape[0], 1))
    higher_error_bar = percentile90 - mean
    higher_error_bar = higher_error_bar.reshape((higher_error_bar.shape[0], 1))
    error_bars = np.concatenate((lower_error_bar, higher_error_bar), axis=1).T

    x_ticks = np.expand_dims((np.arange(len(data) / 2) + 1), axis=1)
    x_ticks = np.concatenate((x_ticks - 0.1, x_ticks + 0.1), axis=1).reshape((len(data),))

    fig, ax = plt.subplots()

    ecolors = [LIGHT_BLUE, BLUE] * 6
    for i in range(len(x_ticks)):
        # errorbar made in a loop to be able
        # to assign different ecolor for different bars
        ax.errorbar(x_ticks[i],
                    mean[i],
                    yerr=np.reshape(error_bars[:,i], (2,1)),
                    fmt='o',
                    markerfacecolor='w',
                    markeredgecolor=GRAY,
                    ecolor=ecolors[i],
                    markersize=MARKERSIZE,
                    elinewidth=ELINEWIDTH,
                    capsize=CAPSIZE,
                    capthick=CAPTHICK
        )
    
    # ax.set_xticklabels(configs)

    ax.set_xlabel(r'\textbf{output of layer}',
                  fontsize=fontsize,
                  color=GRAY)
    ax.set_ylabel(r'\textbf{time to 90\% production}',
                  fontsize=fontsize,
                  color=GRAY)
    ax.grid()

    plt.xticks(fontsize=fontsize, color=ORANGE, weight='bold')
    plt.yticks(fontsize=fontsize, color=ORANGE, weight='bold')

    plt.tight_layout()
    plt.savefig('heart_90fraction_production_per_layer.pdf',
                bbox_inches='tight')
    plt.close()
    
    
if __name__ == '__main__':
    # Enable TEX commands, like \textbf
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
    
    # means, stds, maxes, mins = load_data_averaged(['1', '2', '3', '4', '5'],
    #                                                ['a', 'b'])
    # plot_mean_std(np.array(means), np.array(stds))

    configs, all_data = load_data_convergence(column='order_of_highest_convergence')
    error_bar(
        np.array(all_data),
        sort_by_mean=False,
        out_file='heart_stabilization_time_scaling.pdf',
        ylabel='stabilization time')

    configs, all_data = load_data_convergence(column='output_90fraction_completion')
    error_bar(
        np.array(all_data),
        sort_by_mean=False,
        out_file='heart_90fraction_production_scaling.pdf',
        ylabel='time to 90\% production'
    )

    configs, all_data = load_data_90production_per_layer()
    error_bar_90production_per_layer(np.array(all_data))
