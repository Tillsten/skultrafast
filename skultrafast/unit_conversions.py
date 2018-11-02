from scipy.constants import physical_constants, c

c_cm = c*100


def fs2cm(t):
    return 1/(t*1e-15 * c_cm)

def cm2fs(cm):
    return  1e15/(cm * c_cm)

def nm2cm(nm):
    return 1e7/nm

def cm2nm(cm):
    return 1e7/cm

def cm2eV(cm):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m/100
    return cm/eV_cm

def eV2cm(eV):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m/100
    return eV*eV_cm

def cm2THz(cm):
    return 1/fs2cm(cm)/1e-3

def THz2cm(THz):
    return cm2fs(1e3/THz)