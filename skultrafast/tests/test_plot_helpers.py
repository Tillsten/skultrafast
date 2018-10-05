from skultrafast.plot_helpers import nsf

def test_nsf():
    assert(nsf(0.123)  == '0.12')
    assert(nsf(2.345)  == ' 2.3')
    assert(nsf(5.4324) == '   5')
    assert (nsf(66.43) == '  70')
