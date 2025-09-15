# Multispectral Camera Sim Model


## How to run

```
from main import *
from lib.dataclasses import *

fs_unit = FilterSensorUnit(FilterSpecs(np.array([])), SensorSpecs(np.array([])), np.array([]))

ms_model = MultispectralCameraModel([fs_unit], 'path_to_hs_image_hdr_file')
```
