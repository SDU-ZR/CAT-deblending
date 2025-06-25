import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.cbook import _reshape_2D
import itertools


def my_boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                     autorange=False, percents=[25, 75]):
    '''
    Return statistics computed for boxplot
    '''

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # median
        med = np.percentile(x, 50)
        # med = abs(np.mean(x))
        ## Altered line
        q1, q3 = np.percentile(x, (percents[0], percents[1]))

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats
palette = [
    '#c6bcc0',
    '#1d456d',
    '#759792',
    '#ba9123',
    '#2f6b99',
    '#64532e',
    '#070c13',
    '#a3351a',
    '#0f3849',
    '#c66978',
    '#d5b56b',
    '#19252e',
    '#111b24',
    '#2a5650',
    '#24352b',
    '#162423',
    '#0f1c1b',
    '#1c181e',
    '#34241c',
]

def boxplot_func(df_in, x, y, z,
                 xlim, ylim, ylim2,
                 x_scale,
                 legend,
                 x_label, y_label, y_label_hist, y_label_2,
                 errors=None,
                 legend_remove=False,
                 palette=["#a3351a"],
                 nbins=18,png="0"):
    '''
    Return boxplot figure, median and standard deviation

    Parameters:
    ----------
    df_in: input data
    x, y, z: labels for x, y and z
    xlim, ylim, ylim2: limits for plots
    x_scale: choice 'log' or not
    legend: legend to display
    x_label, y_label, y_label_hist, y_label_2:
    errors: errors to drop if necessary
    legend_remove: boolean to remove the legend
    palette: color palette
    nbins: number of bins to split data
    '''
    median = []
    q1 = []
    q3 = []
    whislo = []
    whishi = []

    import matplotlib as mpl
    mpl.rcdefaults()

    # Drop error if necessary
    if errors is not None:
        df_plot = df_in.drop(errors)
    else:
        df_plot = df_in

    # Drop NaN in dataframe
    df_plot = df_plot.dropna()

    # Define bins
    if x_scale == 'log':
        x_bins = np.geomspace(xlim[0], xlim[1], nbins + 1)
    else:
        x_bins = np.linspace(xlim[0], xlim[1], nbins + 1)

    x_bins[0] -= 1e-5
    x_bins[-1] += 1e-5

    idx = np.digitize(df_plot[x], x_bins) #将“df_plot[x]中的每个元素分配到相应的刻度区间索引中，结果存储在“idx”中”

    # Initialize figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [1, 3, 1]})
    # plt.subplots_adjust(hspace=0)

    # Top plot: distribution of the parameter
    if x_scale == 'log':
        sns.distplot(np.log10(df_plot[x]), kde=False, ax=axes[0], color='0.5')
        axes[0].set_xlim(np.log10(xlim[0]), np.log10(xlim[1]))
    else:
        sns.distplot(df_plot[x], kde=False, ax=axes[0], color='0.5')
        axes[0].set_xlim(xlim[0], xlim[1])

    axes[0].set_yticks([]) #隐藏第一个子图y轴标签
    axes[0].set_ylabel(y_label_hist) #设置y轴标签
    axes[0].set_xticks([]) #表示隐藏x轴刻度标签
    # axes[0].ax.set_facecolor('lavender')

    # Second plot: boxplot generated with seaborn split as a function of the parameter
    ax = axes[1]
    exp = np.unique(df_plot[z]) #获取df_plot[z]列中惟一的值
    N_exp = len(exp)
    print(exp)
    handles = []
    for ik, key in enumerate(exp):
        print(ik, key)
        stats = {}
        # Compute and save statistics
        for i in range(1, len(x_bins)):
            stats[i] = my_boxplot_stats(df_plot[y][np.logical_and(idx == i, df_plot[z] == key)].values,
                                        whis=[100 * scipy.stats.norm.cdf(-2), 100 * scipy.stats.norm.cdf(2)],
                                        percents=[100 * scipy.stats.norm.cdf(-1), 100 * scipy.stats.norm.cdf(1)])[0]
            median.append(stats[i]['med'])
            q1.append(stats[i]['q1'])
            q3.append(stats[i]['q3'])
            whislo.append(stats[i]['whislo'])
            whishi.append(stats[i]['whishi'])
        # Plot boxplots from our computed statistics
        bp = ax.bxp([stats[i] for i in range(1, len(x_bins))],
                    positions=np.arange(len(x_bins) - 1) + .5 + 0.9 * (ik - (N_exp - 1.) / 2.) / N_exp,
                    widths=0.7 / len(np.unique(df_plot[z])),
                    showfliers=False,
                    patch_artist=True,
                    boxprops={'facecolor': (*mpl.colors.to_rgba(palette[ik])[:3], 0.25), 'edgecolor': palette[ik]})

        handles.append(bp['boxes'][0])

        # Colour the lines in the boxplot blue
        for element in bp.keys():
            if element != 'boxes':
                plt.setp(bp[element], color=palette[ik])

    if not legend_remove:
        ax.legend(handles, legend, frameon=False, loc='upper right', borderpad=0.01, fontsize=10)
    ax.axhline(y=0, c='0.5', zorder=32, lw=0.5)
    ax.set_facecolor('lavender')
    ax.set_xlim(0, len(x_bins) - 1)
    ax.set_ylabel(y_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks([])
    ax = ax.twiny()
    ax.set_xlim(xlim[0], xlim[1])
    if x_scale == 'log':
        ax.set_xscale('log')
    ax.set_xticks([])
    ax.xaxis.tick_bottom()


    # Median of boxplot, as a function of the parameter
    ax = axes[2]
    ax.axhline(y=0, c='0.5', zorder=32, lw=0.5)
    med_array = np.array(median).reshape(N_exp, int(len(median) / N_exp))
    for ik, key in enumerate(exp):
        ax.plot(np.repeat(x_bins, 2)[1:-1], np.repeat(med_array[ik], 2), c=palette[ik])
    ax.set_xlim(xlim[0], xlim[1])
    if x_scale == 'log':
        ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylim(ylim2)
    ax.set_ylabel(y_label_2)
    ax.xaxis.tick_bottom()
    ax.set_facecolor('lavender')

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("./figure_5/{}.png".format(png),dpi=300)
    plt.show()

    return fig, median, q1, q3, whislo, whishi



def draw_diffmag_magerr():
    df = pd.read_csv("./test.csv")[0:12000]
    boxplot_func(df, "mag_diff", "mag_err", "class",
                 [-2, 2], [-1.15, 1.15], [-0.25, 0.25],
                 "not",
                 ["12000 total samples"],
                 "Magnitude difference with closest  $\Delta{{mag}_c}$", "Magnitude error $\Delta{mag}$", "P($\Delta_{mag}}$)", "median", palette=["purple"],png="3")

def draw_diffmag_ellipticity_err():
    df = pd.read_csv("./test.csv")[0:12000]
    print(np.mean(df["mag_err"]))
    boxplot_func(df, "mag_diff", "ellipticity_err", "class",
                 [-2, 2], [-0.35, 0.35], [-0.035, 0.035],
                 "not",
                 ["12000 total samples"],
                 "Magnitude difference with closest  $\Delta{{mag}_c}$", "ellipticity error $\Delta{|e|}}$", "P($\Delta_{mag}}$)", "median", palette=["purple"],png="4")
if __name__ == "__main__":
    draw_diffmag_magerr()
