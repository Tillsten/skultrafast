from skultrafast import data_io

def test_load_exmaple():
    wl, t, d = data_io.load_example()
    assert((t.shape[0], wl.shape[0]) == d.shape)
