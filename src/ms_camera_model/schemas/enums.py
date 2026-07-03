"""
Multispectral Camera Model - Enums
==================================

* **Description:** Defines enums used for setting selection
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""

from enum import Enum


class SimulationMode(str, Enum):
    RADIANCE = "radiance"
    REFLECTANCE = "reflectance"
