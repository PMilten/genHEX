from GenHEX_functions import *

# -------------------------------------------------------
# Load setup
# -------------------------------------------------------
class Fluid:
    """Fluid parameters."""
    pass

class HeatExchanger:
    """Geometric, material, and general setup."""
    pass

def evaluate_HEX(setup):
    f1 = setup["f1"]
    f2 = setup["f2"]
    HEX = setup["HEX"]

    print(f"📂 Loaded setup for {f1.fluid} → {f2.fluid}")

    GenHEX(f1, f2, HEX, verbose=False)

    setup = {"f1": f1, "f2": f2, "HEX": HEX}

    return setup
