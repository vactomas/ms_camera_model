# Multispectral Camera Sim Model


## How to run

```
from main import *
import numpy as np

f1 = FilterSpecs(filter_transmission=np.array([]))
f1.import_filter_specs('path_to_xlsx_file')

ms_model = MultispectralCameraModel([f1], 'path_to_hs_image_hdr_file')
```
