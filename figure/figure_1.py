def draw_bar():
    import numpy as np
    import matplotlib.pyplot as plt
    # 星系形态标签
    labels = ['Lenti.', 'Barred', 'Spiral', 'Round', 'In-Between', 'Cigar']
    labels_2 = ['1', '2', '3', '4','5', '6']

    # 星系数量
    all_counts = np.array([15484, 17500, 17162, 11075, 23233, 15836])

    train_counts = np.ceil(all_counts * 0.8)
    test_counts = np.ceil(all_counts * 0.1)
    val_counts = all_counts - test_counts - train_counts

    all_counts_2 = np.ceil(all_counts * 0.5)
    train_counts_2 = np.ceil(all_counts_2 * 0.8)
    test_counts_2 = np.ceil(all_counts_2 * 0.1)
    val_counts_2 = all_counts_2 - test_counts_2 - train_counts_2

    # 计算柱状图位置
    bar_width = 0.3
    index = np.arange(len(labels))

    r3 = [x + bar_width+ 0.05 for x in index]
    # Create the colors
    colors = ['#1f77b4', '#5da5da', '#aec7e8', '#d62728', '#ff9896', '#ffbb78']

    plt.figure(figsize=(10, 8))  # Increased height
    # 绘制柱状图
    plt.bar(index, train_counts, bar_width, label='M_band_train', color=colors[0])
    plt.bar(index, test_counts, bar_width, label='M_band_test', color=colors[1], bottom=train_counts)
    plt.bar(index, val_counts, bar_width, label='M_band_val', color=colors[2], bottom=np.add(train_counts, test_counts))

    plt.bar(r3, train_counts_2, bar_width,label='rgb_train', color=colors[3])
    plt.bar(r3, test_counts_2, bar_width, label='rgb_test' ,color=colors[4], bottom=train_counts_2)
    plt.bar(r3, val_counts_2, bar_width,label='rgb_val' , color=colors[5], bottom=np.add(train_counts_2, test_counts_2))

    # 设置轴标签和标题
    plt.xlabel('Galaxy class', fontsize=18)
    plt.ylabel('number of class', fontsize=18)
    plt.title('Distribution of the number of Galaxy morphology', fontsize=18)

    # 设置x轴刻度标签
    plt.xticks(index+0.175, labels,rotation=45,fontsize=12)
    plt.tight_layout()

    # 添加图例
    plt.legend(fontsize=12)

    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.6))
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.gca().set_facecolor('whitesmoke')
    plt.savefig('./1.png',dpi=100)
    # 显示图形
    plt.show()
draw_bar()