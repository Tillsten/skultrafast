from pathlib import Path
from numpy.testing import assert_almost_equal
import pytest

from skultrafast.data_io import get_twodim_dataset
from skultrafast.quickcontrol import QC2DSpec
from skultrafast.twoD_dataset import TwoDim


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
def two_d_processed(two_d) -> TwoDim:
    two_d = two_d.copy()
    return two_d.select_range((2030, 2200), (2030, 2200))


def test_integrate(two_d_processed):
    two_d_processed.integrate_pump()


def test_cls(two_d_processed):
    two_d = two_d_processed.copy()
    two_d.single_cls(3)
    for m in ['quad', 'fit', 'log_quad', 'skew_fit']:
        two_d.single_cls(3, method=m)


def test_cls_subrange(two_d_processed):
    two_d = two_d_processed.copy()
    two_d.single_cls(3, pr_range=(2140, 2169), pu_range=(2140, 2169))


def test_all_cls(two_d_processed: TwoDim):
    cls_result = two_d_processed.cls()
    for use_const in [True, False]:
        for use_weights in [True, False]:
            cls_result.exp_fit([1], use_const=use_const, use_weights=use_weights)
            cls_result.exp_fit([1, 10], use_const=use_const, use_weights=use_weights)
            assert cls_result.exp_fit_result_ is not None
            cls_result.plot_cls()


def test_diag(two_d_processed: TwoDim):
    d1 = two_d_processed.diag_and_antidiag(3)
    d2 = two_d_processed.diag_and_antidiag(1, offset=0)
    


def test_psa(two_d_processed):
    two_d_processed.pump_slice_amp(3)


def test_savetext(two_d_processed, tmp_path_factory):
    two_d_processed.save_txt(tmp_path_factory.mktemp('data'))


def test_twodplot_contour(two_d_processed):
    two_d_processed.plot.contour(1)
    two_d_processed.plot.contour(1, 3, 5, )
    two_d_processed.plot.elp(1)

def test_bg_correct(two_d_processed: TwoDim):
    tbg = two_d_processed.copy()
    tbg.background_correction((2130, 2160))

    