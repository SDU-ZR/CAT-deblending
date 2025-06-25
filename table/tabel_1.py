if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # A = np.load("D:\PC\Galaxy_deblending\deblender/val_fake_up_img.npy")
    # print(A.shape)
    g_flux = pd.read_csv("./tabel_1/predeblend_r.csv")
    g_sex_flux = pd.read_csv("./tabel_1/blend_central_r.csv")
    g_de_flux = pd.read_csv("./tabel_1/deblend_r.csv")
    g_gan_flux = pd.read_csv("./tabel_1/deblend_r_gan.csv")


    print(len(g_flux))
    def flux2mag(flux, zeropoint=22.5):
        return -2.5 * np.log10(flux) + zeropoint
    def flux2mag_2(flux, zeropoint=25.08):
        return flux
    def mean_absolute_percentage_error(y_true, y_pred):
        diff = np.abs(
            (y_true - y_pred) /
            np.clip(np.abs(y_true), np.finfo(float).eps, None)
        )
        return np.round(100. * np.nanmean(diff, axis=-1), 2)

    def suppple_ax(mag, true_mag):
        dmagabs = abs(mag - true_mag)

        Delta_mag = np.mean(dmagabs)
        sigma_mag = np.std(dmagabs)

        mape = mean_absolute_percentage_error(np.asarray(true_mag), np.asarray(mag))

        return Delta_mag, sigma_mag, mape


    # magnitude 比较
    Delta_mag_sex, sigma_mag_sex, mape_mag_sex = suppple_ax(
        flux2mag(g_flux["flux"][:]), flux2mag(g_sex_flux["flux"][:])
    )
    Delta_mag_de, sigma_mag_de, mape_mag_de = suppple_ax(
        flux2mag(g_flux["flux"][:]), flux2mag(g_de_flux["flux"][:])
    )
    Delta_mag_gan, sigma_mag_gan, mape_mag_gan = suppple_ax(
        flux2mag(g_flux["flux"][:]), flux2mag(g_gan_flux["flux"][:])
    )


    # ellipticity 比较
    Delta_ellipticity_sex, sigma_ellipticity_sex, mape_ellipticity_sex = suppple_ax(
        g_flux["ellipticity"], g_sex_flux["ellipticity"]
    )
    Delta_ellipticity_de, sigma_ellipticity_de, mape_ellipticity_de = suppple_ax(
        g_flux["ellipticity"], g_de_flux["ellipticity"]
    )
    Delta_ellipticity_gan, sigma_ellipticity_gan, mape_ellipticity_gan = suppple_ax(
        g_flux["ellipticity"], g_gan_flux["ellipticity"]
    )

    from tabulate import tabulate

    table_data = [
        ["SExtractor", f"{mape_mag_sex:.2f}%", f"{Delta_mag_sex:.2f}", f"{Delta_ellipticity_sex:.4f}",f"----",f"----"],
        ["GAN", f"{mape_mag_gan:.2f}%", f"{Delta_mag_gan:.2f}", f"{Delta_ellipticity_gan:.4f}",f"{0.96:.2f}",f"{42.65:.2f}"],
        ["CAT-deblender", f"{mape_mag_de:.1f}%", f"{Delta_mag_de:.2f}",
         f"{Delta_ellipticity_de:.4f}",f"{0.99:.2f}",f"{48.79:.2f}"]
    ]

    headers = ["Deblender", "Mag_mape", "Δmag", "Δellipticity","SSIM","PSNR"]

    print(tabulate(table_data, headers=headers, tablefmt="github"))


