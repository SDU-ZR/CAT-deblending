import os
import sep
import numpy as np
import pylab as plt
import pandas as pd

X_CENTER = 64
Y_CENTER = 64
ZP_SEXTRACTOR = 25.67
SEXCONFIG = {
    "hot": {
        "final_area": 6,
        "final_threshold": 4,
        "final_cont": 0.0001,
        "final_nthresh": 64,
    },
    "cold": {
        "final_area": 10,
        "final_threshold": 5,
        "final_cont": 0.01,
        "final_nthresh": 64,
    },
    "mine": {
        "final_area": 10,
        "final_threshold": 4,
        "final_cont": 0.0001,
        "final_nthresh": 64,
    }
}
def run_sextractor(image, background, config):
    return sep.extract(
        image,
        config["final_threshold"],
        err=background.globalrms,
        minarea=config["final_area"],
        deblend_nthresh=config["final_nthresh"],
        deblend_cont=config["final_cont"],
        segmentation_map=True,
    )

def plot_galaxy_with_markers(idx, image, segmap, gal_cen, gal_comp=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.scatter(gal_cen["x"], gal_cen["y"], marker="x", color="red")
    if gal_comp is not None:
        plt.scatter(gal_comp["x"], gal_comp["y"], marker="o")
    plt.subplot(1, 2, 2)
    plt.imshow(segmap)
    plt.savefig(f"./{idx}.png")
    plt.close()

def analyse_single(single_source_image):
    # Compute the background
    bkg = sep.Background(single_source_image)
    single_source_image = single_source_image - bkg
    # Run detection with the 'cold' strategy
    source, segmap = run_sextractor(single_source_image, bkg, SEXCONFIG["cold"])
    n_detect_cold = len(source["x"])
    n_detect_hot = 0

    if n_detect_cold == 0:
        # Rerun SExtractor with  the 'hot' stratefy
        source, segmap = run_sextractor(single_source_image, bkg, SEXCONFIG["hot"])
        n_detect_hot = len(source["x"])

    n_detections = max(n_detect_cold, n_detect_hot)

    if n_detections == 0:
        return {}

    positions = np.hypot(
        source["x"] - X_CENTER,
        source["y"] - Y_CENTER
    )
    indx = positions.argsort().tolist()
    id_central = indx.pop(0)

    result = {}
    result["n_sources"] = n_detections

    if n_detections != 0:
        # Once the central galaxy is found, order the remaining galaxies by flux
        flux_indx = np.argsort(source[indx]["flux"])
        # id_companion = flux_indx[-1]

        result["flux_central"] = source[id_central]["flux"]
        result["a"] = source[id_central]["a"]
        result["b"] = source[id_central]["b"]

        # result["flux_companion"] = source[id_companion]["flux"]


        plot_galaxy_with_markers(1, single_source_image, segmap,
                                     source[id_central])
    else:
        # If a single object is detected even with the 'hot' strategy
        # assign that galaxy to the central or the companion depending
        # on the relative distance to it

        # x_companion = X_CENTER + full_cat["shift_x"][idx]
        # y_companion = Y_CENTER + full_cat["shift_y"][idx]

        # dist_to_companion = np.hypot(
        #     source["x"][0] - x_companion,
        #     source["y"][0] - y_companion
        # )
        dist_to_center = np.hypot(
            source["x"][0] - X_CENTER,
            source["y"][0] - Y_CENTER
        )
        # # if dist_to_companion > dist_to_center:
        # if dist_to_center < 0.5 * cat["g1_rad"][idx]:
        #     result["flux_central"] = source[id_central]["flux"]
        # else:
        #     result["flux_companion"] = source[id_central]["flux"]
    return result
def analyse_sources(k):
    datadir = os.getenv("COINBLEND_DATADIR")
    """
    CAT-deblender
    """
    data_blends = np.load("./val_fake_up_img.npy")
    """
    GAN
    """
    # data_blends = np.load("../val_fake_up_img_contrast.npy")
    # fluxes_cat = np.load(f"{datadir}/{sample}_flux.npy")
    # data_blends = data_blends.transpose((0,3,1,2))
    data_blends = np.array(data_blends[:,k,1,:,:])
    # data_blends = data_blends.byteswap().newbyteorder()

    n_sources = np.empty(len(data_blends), dtype=np.uint8)
    flux_central_estimated = np.empty(len(data_blends), dtype=np.float64)
    ellipticity = np.empty(len(data_blends), dtype=np.float64)
    # flux_companion_estimated = np.empty(len(data_blends), dtype=np.float)


    for id_blend, image in enumerate(data_blends):
        result = analyse_single(image)

        flux_central_estimated[id_blend] = result.get("flux_central", np.nan)
        a = result.get("a", np.nan)
        b = result.get("b", np.nan)
        print(a,b)
        if a!=np.nan and b!=np.nan:
            ellipticity[id_blend] = (a-b)/(a+b)
        else:
            ellipticity[id_blend] = np.nan
        # flux_companion_estimated[id_blend] = result.get("flux_companion", np.nan)
        n_sources[id_blend] = result.get("n_sources", 0)
        print("           -------------已经完成第{}组数据------------".format(id_blend))

    return n_sources, flux_central_estimated,ellipticity

def main():
    df_preblend = pd.DataFrame(columns=['file_name','flux','ellipticity'], dtype='float64')
    df_blend = pd.DataFrame(columns=['file_name','flux','ellipticity'], dtype='float64')
    df_deblend = pd.DataFrame(columns=['file_name','flux','ellipticity'], dtype='float64')


    n_sources_2, g2_sex ,g2_ep= analyse_sources(2)
    n_sources_1, g1_sex ,g1_ep= analyse_sources(0)
    n_sources_3, g3_sex ,g3_ep= analyse_sources(1)

    for i,g_flux in enumerate(g3_sex):
        df_preblend.loc[len(df_preblend)] = [i, g1_sex[i],g1_ep[i]]
        df_deblend.loc[len(df_deblend)] = [i, g2_sex[i],g2_ep[i]]
        df_blend.loc[len(df_blend)] = [i, g3_sex[i],g3_ep[i]]

    df_preblend.to_csv("./figure_8/predeblend_r.csv", index=False)
    df_blend.to_csv("./figure_8/blend_central_r.csv", index=False)
    df_deblend.to_csv("./figure_8/deblend_r.csv",index=False)
    # df_blend.to_csv("./companion_galaxy_flux_r.csv", index=False)

if __name__ == "__main__":
    version = "afterscreening"
    sample = "test"
    fig = False
    main()

    import pandas as pd
    import numpy as np
    df_data = pd.DataFrame(columns=['file_name', 'S/N', 'ellipticity_err', 'mag_err', "class"], dtype='float64')
    import torch
    df_preblend = pd.read_csv("./predeblend_r.csv")
    df_deblend = pd.read_csv("./deblend_r.csv")
    data = np.load("../val_fake_up_img.npy")
    # df_companion = pd.read_csv("./companion_galaxy_flux_r.csv")
    # df_rate = pd.read_csv("./blend_rate.csv")
    def flux2mag(flux, zeropoint=25):
        return -2.5 * np.log10(flux) + zeropoint
    def calculate_psnr(img1, img2):
        return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))

    for i in range(10000):
        img1 = data[i, 0, 1, :, :]
        img2 = data[i, 2, 1, :, :]
        psnr = calculate_psnr(img1, img2)
        ellipticity_err = df_preblend["ellipticity"][i] - df_deblend["ellipticity"][i]
        mag_err = flux2mag(df_preblend["flux"][i]) - flux2mag(df_deblend["flux"][i])
        # mag_diff = flux2mag(df_preblend["flux"][i]) - flux2mag(df_companion["flux"][i])
        df_data.loc[len(df_data)] = [i, psnr, ellipticity_err, mag_err, 0]
        print("      ---------------已经完成第{}组数据-----------------".format(i))
    df_data.to_csv("./figure_8/test.csv", index=False)





