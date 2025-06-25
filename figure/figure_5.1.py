import numpy as np


def psnr_show(psnr_values_list_copy):
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置PSNR区间
    num_bins = int((45 - 18) / 0.5)
    bins = [i for i in np.arange(18, 45, 0.5)]
    psnr_values_list = psnr_values_list_copy.copy()
    for i in range(len(psnr_values_list)):
        for d in range(i):
            psnr_values_list[i] = np.concatenate((psnr_values_list[i], psnr_values_list_copy[d]))

            # 统计每个组内的PSNR值数量
    hist_list = [np.histogram(group_values, bins=bins)[0] for group_values in psnr_values_list]

    # 绘制折线图
    fig, ax = plt.subplots()
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, hist in enumerate(hist_list):
        x = bins[1:]
        ax.plot(x, hist, color=colors[i])

        if i > 0:
            prev_hist = hist_list[i - 1]
            prev_color = colors[i - 1]
            ax.fill_between(x, prev_hist, hist, color=prev_color, alpha=0.3, label=f' Galaxy calss {i + 1}')
        if i == 0:
            ax.fill_between(x, hist_list[-1], color=colors[-1], alpha=0.3, label=f' Galaxy calss 1')

        if i == 5:
            median_psnr = np.median(psnr_values_list[i])
            mean_psnr = np.mean(psnr_values_list[i])
            ax.axvline(x=median_psnr, color='k', label='Median', linestyle='--')
            ax.axvline(x=mean_psnr, color='red', label='Mean', linestyle='--')
            ax.grid(color='white')
            ax.xaxis.grid(True, linewidth=0.5, which='major', color='white')
            ax.set_facecolor('lavender')
    ax.set_xlabel('PSNR(dB)')
    ax.set_ylabel('Number of Galaxy')
    ax.legend()
    fig.text(0.5, 0.005, 'A', ha='center', color='red', fontsize=24)
    plt.tight_layout()
    plt.savefig("./figure_5/psnr.png", dpi=300)
    # 显示图像
    plt.show()


def ssim_show(psnr_values_list_copy):
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置PSNR区间
    # num_bins = int((45 - 18) / 0.5)
    bins = [i for i in np.arange(0.5, 1, 0.005)]
    psnr_values_list = psnr_values_list_copy.copy()
    for i in range(len(psnr_values_list)):
        for d in range(i):
            psnr_values_list[i] = np.concatenate((psnr_values_list[i], psnr_values_list_copy[d]))

            # 统计每个组内的PSNR值数量
    hist_list = [np.histogram(group_values, bins=bins)[0] for group_values in psnr_values_list]

    # 绘制折线图
    fig, ax = plt.subplots()
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, hist in enumerate(hist_list):
        x = bins[1:]
        ax.plot(x, hist, color=colors[i])

        if i > 0:
            prev_hist = hist_list[i - 1]
            prev_color = colors[i - 1]
            ax.fill_between(x, prev_hist, hist, color=prev_color, alpha=0.3, label=f' Galaxy calss {i + 1}')
        if i == 0:
            ax.fill_between(x, hist_list[-1], color=colors[-1], alpha=0.3, label=f' Galaxy calss 1')

        if i == 5:
            median_psnr = np.median(psnr_values_list[i])
            mean_psnr = np.mean(psnr_values_list[i])
            ax.axvline(x=median_psnr, color='k', label='Median', linestyle='--')
            ax.axvline(x=mean_psnr, color='red', label='Mean', linestyle='--')
            ax.grid(color='white')
            ax.xaxis.grid(True, linewidth=0.5, which='major', color='white')
            ax.set_facecolor('lavender')

    ax.set_xlabel('SSIM')
    ax.set_ylabel('Number of Galaxy')
    ax.legend()
    fig.text(0.5, 0.01, 'B', ha='center', color='red', fontsize=24)
    plt.tight_layout()
    plt.savefig("./figure_5/ssim.png", dpi=300)
    # 显示图像
    plt.show()
def calculate_statistics(data_list):
    # 均值
    mean = sum(data_list) / len(data_list)

    # 方差
    variance = sum((x - mean) ** 2 for x in data_list) / len(data_list)

    # 中位数
    sorted_data = sorted(data_list)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    # 最大值和最小值
    max_value = max(data_list)
    min_value = min(data_list)

    return mean, variance, median, max_value, min_value

ssims = []
psnrs = []
for sample_size,index in enumerate([750,1000,750,500,1250,750]):
    new_psnr = []
    new_ssim = []
    d = 6 - sample_size
    psnr = np.load("./Origin/psnr_{}.npy".format(d))
    ssim = np.load("./Origin/ssim_{}.npy".format(d))

    sorted_psnr = np.sort(psnr)
    sorted_ssim = np.sort(ssim)
    step = len(sorted_psnr) // index
    for i in range(0, len(sorted_psnr), step):
        new_psnr.append(sorted_psnr[i])
        new_ssim.append(sorted_ssim[i])

    ssims.append(new_ssim)
    psnrs.append(new_psnr)


sssim = [item for small_list in ssims for item in small_list]
ppsnr = [item for small_list in psnrs for item in small_list]
mean, variance, median, max_value, min_value = calculate_statistics(sssim)
print(mean, variance, median, max_value, min_value)
mean, variance, median, max_value, min_value = calculate_statistics(ppsnr)
print(mean, variance, median, max_value, min_value)
psnr_show(psnrs)
ssim_show(ssims)