import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs(
        (y_true - y_pred) /
        np.clip(np.abs(y_true), np.finfo(float).eps, None)
    )

    return np.round(100. * np.nanmean(diff, axis=-1), 2)
import matplotlib.colors as mcl

paper_palette = [
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

gal1_cmap = mcl.LinearSegmentedColormap.from_list('paper_blue', ((1,1,1), '#1d456d'), N=256)
gal2_cmap = mcl.LinearSegmentedColormap.from_list('paper_brown', ((1,1,1), '#64532e'), N=256)
img_cmap = mcl.LinearSegmentedColormap.from_list('paper_BlBr', ['#1d456d', (0.8,0.8,0.8), '#64532e'])

PALETTE = paper_palette
sns.set_palette(PALETTE)
sns.set_context("paper")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)
plt.rc("font", size=20)
plt.rc("xtick", labelsize='LSM')
plt.rc("ytick", labelsize='LSM')

ZP_SEX = 25.67
ZP_NN = 25.96
XMIN = 15
XMAX = 20
YMIN = 15
YMAX = 20
XTEXT = 15.5
YTEXT = 19.5
TEXTSIZE = 24
PLOT_XTICKS = list(range(XMIN+1, XMAX))
PLOT_YTICKS = list(range(YMIN+1, YMAX))
HIST_XTICKS = [-1.0, 0.0, 1.0]
def plot_ax(ax, title, mag, true_mag, color, ylabel=False):
    dmagabs = abs(mag - true_mag)
    dmag = mag - true_mag

    ax.set_title(title, fontsize=26, fontstyle='italic')
    ax.plot([XMIN, XMAX], [YMIN, YMAX], lw=1, ls="--", color='gray')
    ax.scatter(true_mag, mag, s=6, color=color, alpha=0.15)
    ax.set_xticks(PLOT_XTICKS)
    ax.set_xlabel("Magnitude isolated", fontsize=22)
    if ylabel:
        ax.set_yticks(PLOT_YTICKS)
        ax.set_ylabel("Magnitude blended recovered", fontsize=22)
    else:
        ax.set_yticks([])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    ax.text(
        XTEXT, YTEXT,
        r"$\overline{\Delta_{mag}}$" + " = {0:.3f}".format(np.mean(dmagabs)),
        fontsize=TEXTSIZE,
    )
    ax.text(
        XTEXT, YTEXT - 0.5,
        r"${\sigma_{mag}}$" + " =  {0:.3f}".format(np.std(dmagabs)),
        fontsize=TEXTSIZE,
    )

    n = (mag[np.abs(dmag) > 0.75]).shape[0]
    ax.text(
        XTEXT, YTEXT - 1,
        "outliers" + " = {0:2.1f}%".format(n / (dmagabs).shape[0] * 100),
        fontsize=TEXTSIZE,
    )
    mape = mean_absolute_percentage_error(np.asarray(true_mag), np.asarray(mag))
    print(mape)
    ax.text(
        XTEXT, YTEXT - 1.5,
        r"MAPE" + " = {0:2.1f}%".format(mape),
        fontsize=TEXTSIZE,
    )

    in_ax = inset_axes(ax, width="37%", height="37%", loc=4, borderpad=4)

    sns.distplot(
        dmag,
        hist=True,
        kde=True,
        # norm_hist=True,
        bins=int(180 / 5),
        color=color,
        hist_kws={"edgecolor": color},
        kde_kws={"linewidth": 4},
        ax=in_ax,
    )
    plt.title(r"$\Delta_{\rm mag}$", fontsize=24)#, fontsize=26)
    plt.xlim(-2, 2)
    # plt.ylim(0, 5)
    plt.xticks(HIST_XTICKS, fontsize=18)
    plt.yticks([])
    for loc in ['left', 'top', 'right','right']:
        in_ax.spines[loc].set_visible(True)

def flux2mag(flux, zeropoint):
    return -2.5 * np.log10(flux) + zeropoint
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    g_flux = pd.read_csv("figure_8/predeblend_r.csv")
    g_sex_flux = pd.read_csv("figure_8/blend_central_r.csv")
    g_de_flux = pd.read_csv("figure_8/deblend_r.csv")
    fig5 = plt.figure(figsize=(30,10))
    fig5.set_size_inches(8.75 * 4, 8.75 * 3)
    ax1 = fig5.add_subplot(3, 4, 1)
    ax5 = fig5.add_subplot(3, 4, 5)
    ax6 = fig5.add_subplot(3, 4, 6)
    ax7 = fig5.add_subplot(3, 4, 7)
    ax8 = fig5.add_subplot(3, 4, 8)
    plot_ax(ax1, "4000_double_gal_mag\n(after deblended by Sextractor)", flux2mag(g_flux["flux"][0:4000],22.5), flux2mag(g_sex_flux["flux"][0:4000],22.5), PALETTE[4], ylabel=True)
    ax2 = fig5.add_subplot(3, 4, 2)
    plot_ax(ax2, "4000_triple_gal_mag\n"
                 "(after deblended by Sextractor)", flux2mag(g_flux["flux"][4000:8000],22.5), flux2mag(g_sex_flux["flux"][4000:8000],22.5), PALETTE[4], ylabel=True)
    ax3 = fig5.add_subplot(3, 4, 3)
    plot_ax(ax3, "4000_quadruple_gal_mag\n(after deblended by Sextractor)", flux2mag(g_flux["flux"][8000:12000],22.5), flux2mag(g_sex_flux["flux"][8000:12000],22.5), PALETTE[4], ylabel=True)
    ax4 = fig5.add_subplot(3, 4, 4)
    plot_ax(ax4, "12000_all_gal_mag\n(after deblended by Sextractor)", flux2mag(g_flux["flux"][0:12000],22.5), flux2mag(g_sex_flux["flux"][0:12000],22.5), PALETTE[4], ylabel=True)
    plot_ax(ax5, "4000_double_gal_mag\n(after deblended by CAT-deblender)", flux2mag(g_flux["flux"][0:4000],22.5), flux2mag(g_de_flux["flux"][0:4000],22.5), PALETTE[7], ylabel=True)
    plot_ax(ax6, "4000_triple_gal_mag\n(after deblended by CAT-deblender)", flux2mag(g_flux["flux"][4000:8000],22.5), flux2mag(g_de_flux["flux"][4000:8000],22.5), PALETTE[7], ylabel=True)
    plot_ax(ax7, "4000_quadruple_gal_mag\n(after deblended by CAT-deblender)", flux2mag(g_flux["flux"][8000:12000], 22.5),
            flux2mag(g_de_flux["flux"][8000:12000], 22.5), PALETTE[7], ylabel=True)
    plot_ax(ax8, "12000_all_gal_mag\n(after deblended by CAT-deblender)", flux2mag(g_flux["flux"][0:12000], 22.5),
            flux2mag(g_de_flux["flux"][0:12000], 22.5), PALETTE[7], ylabel=True)

    # fig5.subplots_adjust(wspace=0, top=0.95, left=0.05, right=0.99, bottom=0.05, hspace=0.25)
    ax9 = fig5.add_subplot(3, 4, 9)
    ax9.set_position([0.125, 0.1, 0.775, 0.2])
    ax9.imshow(plt.imread("./figure_8/1_1.png"))

    ax10 = fig5.add_subplot(3, 4, 10)
    ax10.imshow(plt.imread("./figure_8/1_2.png"))

    ax11 = fig5.add_subplot(3, 4, 11)
    ax11.imshow(plt.imread("./figure_8/1_3.png"))

    ax12 = fig5.add_subplot(3, 4, 12)
    ax12.imshow(plt.imread("./figure_8/1_4.png"))

    for ax in [ax9, ax10,ax11,ax12]:
        ax.axis('off')
    fig5.subplots_adjust(wspace=0, top=0.95, left=0.03, right=0.99, bottom=0.00, hspace=0.25)

    plt.show()
    fig5.savefig("./8.png")


