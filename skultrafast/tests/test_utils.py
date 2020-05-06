from skultrafast.utils import pfid_r4, pfid_r6
import numpy as np 

def test_pfid():
    t = np.linspace(0, 10, 100)
    fre = np.linspace(900, 1200, 64)
    y1 = pfid_r4(t, fre, [1000, 1100], [2, 1])    
    y2 = pfid_r6(t, fre, [1000], [1015],  [2])