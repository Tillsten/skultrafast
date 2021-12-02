import pytest

from skultrafast.quickcontrol import QC2DSpec
from skultrafast.data_io import get_twodim_dataset
from skultrafast.twoD_dataset import TwoDim
from pathlib import Path


@pytest.fixture(scope='session')
def datadir2d(tmp_path_factory):
    p = get_twodim_dataset()
    return p


@pytest.fixture(scope='session')
def two_d(datadir2d) -> TwoDim:
    info = list(Path(datadir2d).glob('*320.info'))
    qc = QC2DSpec(info[0])
    ds = qc.make_ds()["iso"]
    return ds


def test_select(two_d):
    two_d = two_d.copy()
    two_d = two_d.select_range((2030, 2200), (2030, 2200))

    assert two_d.pump_wn.min() > 2030
    assert two_d.pump_wn.size > 0
    assert two_d.pump_wn.max() < 2200

    assert two_d.probe_wn.min() > 2030
    assert two_d.probe_wn.size > 0
    assert two_d.probe_wn.max() < 2200


    two_d = two_d.select_t_range(1)

    assert two_d.t.min() > 1


@pytest.fixture(scope='session')
def two_d_processed(two_d):
    two_d = two_d.copy()
    return two_d.select_range((2030, 2200), (2030, 2200))


def test_integrate(two_d_processed):
    two_d_processed.integrate_pump()


def test_cls(two_d_processed):
    two_d = two_d_processed.copy()
    two_d.single_cls(3)
    for m in ['quad', 'fit', 'log_quad']:
        two_d.single_cls(3, method=m)
    cls_result = two_d.cls()
    cls_result.plot_cls()


def test_diag(two_d_processed):
    two_d_processed.diag_and_antidiag(3)


def test_psa(two_d_processed):
    two_d_processed.pump_slice_amp(3)
