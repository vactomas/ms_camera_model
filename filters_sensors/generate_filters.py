import numpy as np
import pandas

cwl_bands = [475, 560, 668, 717, 842]
bandwidths = [32, 27, 16, 12, 57]
band_names = ["Blue", "Green", "Red", "Red_Edge", "NIR"]

wavelengths = np.arange(250, 1151, 1)
shape_coefficient = 10

for i_band, band_name in enumerate(band_names):

    half_width = bandwidths[i_band] / 2

    transmittance = np.exp(-np.power(np.abs(wavelengths - cwl_bands[i_band]) / half_width, shape_coefficient))

    df = pandas.DataFrame({"Wavelengths": wavelengths, "Transmittance": transmittance})

    df.to_excel(f"./AltumPT_{band_name}.xlsx", index=False)
