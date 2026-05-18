import numpy as np
import pandas

wavelengths = np.arange(250, 1151, 1)
shape_coefficient = 10

for cwl in range(450, 900, 40):
    for fwhm in range(10, 110, 30):
        half_width = fwhm / 2
        transmittance = np.exp(-np.power(np.abs(wavelengths - cwl) / half_width, shape_coefficient))

        df = pandas.DataFrame({"Wavelengths": wavelengths, "Transmittance": transmittance})

        df.to_excel(f"./custom_filters/Filter_{cwl}-{fwhm}.xlsx", index=False)
