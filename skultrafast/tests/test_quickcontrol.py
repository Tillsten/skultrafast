import pytest
import tempfile
import zipfile
import zipfile_deflate64
from pathlib import Path

from skultrafast.quickcontrol import QC1DSpec, QC2DSpec, parse_str, QCFile
from skultrafast.data_io import get_example_path, get_twodim_dataset


def test_parse():
    assert (parse_str('-8000.000000') == -8000.0)
    assert (parse_str('75') == 75)
    assert (parse_str('TRUE') == True)
    assert (parse_str('FALSE') == False)
    flist = '-8000.000000,-7950.000000,-7900.000000,-7850.000000'
    res = parse_str(flist)
    assert (isinstance(res, list))
    assert (res[0] == -8000)
    assert (len(res) == 4)


@pytest.fixture(scope='session')
def datadir(tmp_path_factory):
    p = get_example_path('quickcontrol')
    tmp = tmp_path_factory.mktemp("data")
    zipfile.ZipFile(p).extractall(tmp)
    return tmp


@pytest.fixture(scope='session')
def datadir2d(tmp_path_factory):
    p = get_twodim_dataset()
    return p


def test_info(datadir):
    qc = QCFile(fname=datadir / '20201029#07')


def test_1d(datadir):
    qc = QC1DSpec(fname=datadir / '20201029#07')
    assert (qc.par_data.shape == qc.per_data.shape)
    assert (qc.par_data.shape[1] == len(qc.t))
    assert (qc.par_data.shape[2] == 128)
    ds = qc.make_pol_ds()
    ds.plot.spec(1)


def test_2d(datadir2d):
    infos = list(Path(datadir2d).glob('*320.info'))
    ds = QC2DSpec(infos[0])
    ds.make_ds()
