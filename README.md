from main import *
import numpy as np

f1 = FilterSpecs(filter_transmission=np.array([]))
f1.import_filter_specs()

ms_model = MultispectralCameraModel([f1], '../test.hdr')
