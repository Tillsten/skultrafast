from skultrafast import data_io
from pathlib import Path


def test_load_exmaple():
    wl, t, d = data_io.load_example()
    assert ((t.shape[0], wl.shape[0]) == d.shape)


def test_path_loader():
    p = data_io.get_example_path('sys_response')
    assert Path(p).exists()
    p = data_io.get_example_path('messpy')
    assert Path(p).exists()


def test_2d_load():
    p = data_io.get_example_path('quickcontrol')
    assert Path(p).exists()


def test_2d_webload():
    a = data_io.get_twodim_dataset()

