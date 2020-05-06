from skultrafast.plot_helpers import nsf

def test_nsf():
    assert(nsf(0.123)  == '0.12')
    assert(nsf(2.345)  == ' 2.3')
    assert(nsf(6.4324) == ' 6.4')
    assert (nsf(66.43) == '  70')
