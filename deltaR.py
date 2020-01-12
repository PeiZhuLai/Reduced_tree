import numpy as np
import math

def deltaR( eta1, phi1, eta2, phi2):
    return math.sqrt((phi1 - phi2) * (phi1 - phi2) + (eta1 - eta2) * (eta1 - eta2))
