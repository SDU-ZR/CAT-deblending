import numpy as np
import pandas as pd
import os
import shutil


def Mor_sure(df):
    df0 = []
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    df_list = [df0, df1, df2, df3, df4]
    df_new = pd.DataFrame(columns=['id', 'type'])

    for index, row in df.iterrows():
        if row['Class1.1'] >= 0.469 and row['Class7.1'] >= 0.5:
            df0.append(row['GalaxyID'])
            df_new.loc[len(df_new)] = [row['GalaxyID'], 0]
        elif row['Class1.1'] >= 0.469 and row['Class7.2'] >= 0.5:
            df1.append(row['GalaxyID'])
            df_new.loc[len(df_new)] = [row['GalaxyID'], 1]
        elif row['Class1.1'] >= 0.469 and row['Class7.3'] >= 0.5:
            df2.append(row['GalaxyID'])
            df_new.loc[len(df_new)] = [row['GalaxyID'], 2]
        elif row['Class1.2'] >= 0.43 and row['Class2.1'] >= 0.602:
            df3.append(row['GalaxyID'])
            df_new.loc[len(df_new)] = [row['GalaxyID'], 3]
        elif row['Class1.2'] >= 0.43 and row['Class2.2'] >= 0.715 and row['Class4.1'] >= 0.619:
            df4.append(row['GalaxyID'])
            df_new.loc[len(df_new)] = [row['GalaxyID'], 4]

    return df_list,df_new

def move_class_data(Galaxy_class,image_names):
    source_folder = "./GalaxyZoo_RGB"
    destination_folder = "./data/{}".format(Galaxy_class)

    # 创建目标文件夹，如果目标文件夹不存在
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    # 循环遍历数组中的所有图片名
    for image_name in image_names:
        # 构建源文件的完整路径和目标文件的完整路径
        image_name = str(int(image_name))
        source_path = os.path.join(source_folder, image_name+'.jpg')
        destination_path = os.path.join(destination_folder, image_name+'.jpg')

        # 复制文件
        shutil.copy(source_path, destination_path)

import pandas as pd

def select_galaxy(df):
    df_new = pd.DataFrame(columns=['ID', 'type', 'ra', 'dec'], dtype='int64')

    for index, row in df.iterrows():
        # Class 0: Lenticular
        if (row['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.40 and
            row['t02_edgeon_a04_yes_weighted_fraction'] > 0.45):
            df_new.loc[len(df_new)] = [row['dr7objid'], 0, row['ra'], row['dec']]

        # Class 1: Barred spiral
        elif (row['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.40 and
              row['t02_edgeon_a05_no_weighted_fraction'] > 0.55 and
              row['t03_bar_a06_bar_weighted_fraction'] > 0.45 and
              row['t04_spiral_a08_spiral_weighted_fraction'] > 0.50):
            df_new.loc[len(df_new)] = [row['dr7objid'], 1, row['ra'], row['dec']]

        # Class 2: Spiral
        elif (row['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.45 and
              row['t02_edgeon_a05_no_weighted_fraction'] > 0.55 and
              row['t03_bar_a07_no_bar_weighted_fraction'] > 0.55 and
              row['t04_spiral_a08_spiral_weighted_fraction'] > 0.50):
            df_new.loc[len(df_new)] = [row['dr7objid'], 2, row['ra'], row['dec']]

        # Class 3: Completely round smooth
        elif (row['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.40 and
              row['t07_rounded_a16_completely_round_weighted_fraction'] > 0.48):
            df_new.loc[len(df_new)] = [row['dr7objid'], 3, row['ra'], row['dec']]

        # Class 4: In between smooth
        elif (row['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.40 and
              row['t07_rounded_a17_in_between_weighted_fraction'] > 0.55):
            df_new.loc[len(df_new)] = [row['dr7objid'], 4, row['ra'], row['dec']]

        # Class 5: Cigar-shaped smooth
        elif (row['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.40 and
              row['t07_rounded_a18_cigar_shaped_weighted_fraction'] > 0.45):
            df_new.loc[len(df_new)] = [row['dr7objid'], 5, row['ra'], row['dec']]




    return df_new



if __name__ == "__main__":

    """
    筛选数据
    """
    # df = pd.read_csv('./zoo2MainSpecz.csv')
    # df_new = select_galaxy(df)
    # df_new.to_csv("classified_galaxies.csv", index=False)

    from tabulate import tabulate

    headers = ["Class", "Galaxy", "Tasks", "Selected responses", "Thresholds", "Sample"]

    table_data = [
        [0, "Lenticular", "T01, T02", "Class 1.2, Class 2.1",
         "f_features/disc > 0.40\nf_edge-on, yes > 0.45", 15484],

        [1, "Barred spiral", "T01, T02, T03, T04",
         "Class 1.2, Class 2.2, Class 3.1, Class 4.1",
         "f_features/disc > 0.40\nf_edge-on, no > 0.55\nf_a bar feature, yes > 0.45\nf_spiral, yes > 0.50", 17500],

        [2, "Spiral", "T01, T02, T03, T04",
         "Class 1.2, Class 2.2, Class 3.2, Class 4.1",
         "f_features/disc > 0.45\nf_edge-on, no > 0.55\nf_a bar feature, no > 0.55\nf_spiral, yes > 0.5", 17162],

        [3, "Completely round smooth", "T01, T07",
         "Class 1.1, Class 7.1",
         "f_smooth > 0.40\nf_completely round > 0.48", 11075],

        [4, "In between smooth", "T01, T07",
         "Class 1.1, Class 7.2",
         "f_smooth > 0.40\nf_in between > 0.55", 23233],

        [5, "Cigar-shaped smooth", "T01, T07",
         "Class 1.1, Class 7.3",
         "f_smooth > 0.40\nf_cigar shaped > 0.45", 15836],
    ]

    print(tabulate(table_data, headers=headers, tablefmt="github"))








