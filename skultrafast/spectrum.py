from attr import dataclass, define, evolve, field
import numpy as np

from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

def _default_fit_style():
    return {'color': 'k', 'linewidth': 1}

def _default_line_style():
    return {}

@dataclass
class PlotOptions:
    xlabel: str = 'Wavenumber [cm-1]'
    ylabel: str = 'Absorption [OD]'
    fit_style: dict = field(factory=_default_fit_style)
    line_style: dict = field(factory=_default_line_style)

@dataclass
class Spectrum1D:
    x: np.ndarray
    y: np.ndarray
    y_baseline: Optional[np.ndarray] = None
    plot_ops: PlotOptions = field(factory=PlotOptions)

    def copy(self):
        cpy = evolve(self)
        cpy.x = self.x.copy()
        cpy.y = self.y.copy()

    def select(self, low: float = -np.inf, high: float = np.inf, invert=False):
        """
        Selects a subrange of the spectrum. The range is defined by the
        low and high values. If invert is True, the outside of the range is selected.
        """
        idx = (self.x >= low) & (self.x <= high)
        if invert:
            idx = ~idx
        return Spectrum1D(x=self.x[idx], y=self.y[idx])

    def est_poly_baseline(self, poly_deg, region: Optional[Tuple[float, float]]=None,
                          exclude: List[Tuple[float, float]]=[]):
        if region is None:
            idx = np.ones(self.x.size, dtype=bool)
        else:
            high = max(*region)
            low = min(*region)
            idx = np.where((self.x >= low) & (self.x <= high))
        assert isinstance(idx, np.ndarray)
        for r in exclude:
            high = max(*r)
            low = min(*r)
            idx[((self.x >= low) & (self.x <= high))] = False
        x, y = self.x[idx], self.y[idx]
        coefs = np.polyfit(x, y, deg=poly_deg)
        yfit = np.polyval(coefs, self.x)
        assert isinstance(yfit, np.ndarray)
        self.y_baseline = yfit

    def plot(self, ax: Optional[plt.Axes]=None):
        if ax is None:
            ax = plt.gca() # type: plt.Axes
        ax.plot(self.x, self.y)
        if self.y_baseline is not None:
            ax.plot(self.x, self.y_baseline)
