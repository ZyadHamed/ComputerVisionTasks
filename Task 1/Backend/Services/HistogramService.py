import io

import matplotlib
matplotlib.use("Agg")  # headless backend, required on a server
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class Histogram:
    LEVELS = 256  # intensity levels 0..255

    def __init__(self, x):  # x is the input image
        self.x = np.asarray(x)
        # last dim == 3 and at least 2 dims (H, W) -> RGB
        self.is_rgb = self.x.ndim >= 2 and self.x.shape[-1] == 3
        self.y = None

    # ---------- helpers ----------
    def _get_y(self, y=None):
        """Use the given y, or fall back to self.y (computing it if needed)."""
        if y is None:
            if self.y is None:
                self.frequency()
            y = self.y
        return np.asarray(y)

    def _per_channel(self, func, arr):
        """Apply func to a 1D array, or to each row of a (3, 256) array."""
        if arr.ndim == 2:
            return np.array([func(row) for row in arr])
        return func(arr)

    # ---------- frequency ----------
    def frequency(self):
        if self.is_rgb:
            channels = [self.x[..., c].ravel() for c in range(3)]
        else:
            channels = [self.x.ravel()]

        result = []
        for ch in channels:
            counts = Counter(ch.tolist())
            result.append(np.array([counts.get(level, 0) for level in range(self.LEVELS)]))

        self.y = np.array(result) if self.is_rgb else result[0]
        return self.y

    # ---------- pdf / cdf (from scratch) ----------
    def _pdf_1d(self, y):
        n = 0
        for freq in y:
            n += freq

        pdf = np.zeros(self.LEVELS)
        for level in range(self.LEVELS):
            pdf[level] = y[level] / n
        return pdf

    def _cdf_1d(self, pdf):
        cdf = np.zeros(self.LEVELS)
        running_total = 0.0
        for level in range(self.LEVELS):
            running_total += pdf[level]
            cdf[level] = running_total
        return cdf

    def pdf(self, y=None):
        """pdf for every level. Shape (256,) or (3, 256) for RGB."""
        y = self._get_y(y)
        return self._per_channel(self._pdf_1d, y)

    def cdf(self, y=None):
        """cdf for every level. Shape (256,) or (3, 256) for RGB."""
        return self._per_channel(self._cdf_1d, self.pdf(y))

    # ---------- equalization ----------
    def equalized_level(self, y=None):
        """x_new[level] = round(cdf[level] * (L - 1)), with L = 256 levels."""
        cdf = self.cdf(y)
        return np.rint(cdf * (self.LEVELS - 1)).astype(int)

    def _equalize_1d(self, y, x_new):
        y_eq = np.zeros(self.LEVELS, dtype=int)
        for old_level, new_level in enumerate(x_new):
            y_eq[new_level] += y[old_level]  # repeated new levels get summed
        return y_eq

    def equalize_histogram(self, y, x_new):
        """Returns x_equalized (0..255) and y_equalized."""
        y, x_new = np.asarray(y), np.asarray(x_new)
        x_equalized = np.arange(self.LEVELS)

        if y.ndim == 2:
            y_equalized = np.array([self._equalize_1d(yr, xr) for yr, xr in zip(y, x_new)])
        else:
            y_equalized = self._equalize_1d(y, x_new)
        return x_equalized, y_equalized

    def equalize_image(self):
        """Apply the equalization mapping to the actual pixels (uint8 image)."""
        x_new = self.equalized_level(self._get_y())  # (256,) or (3, 256)
        img = self.x.astype(np.intp)

        if self.is_rgb:
            out = np.stack([x_new[c][img[..., c]] for c in range(3)], axis=-1)
        else:
            out = x_new[img]
        return out.astype(np.uint8)

    # ---------- plotting (returns PNG bytes) ----------
    def _plot(self, data, title="", ylabel="Frequency", kind="bar"):
        """kind = 'bar' (histogram / pdf) or 'line' (cdf). Returns PNG bytes."""
        levels = np.arange(self.LEVELS)

        def draw(ax, values, color):
            if kind == "bar":
                ax.bar(levels, values, width=1.0, color=color)
            else:
                ax.plot(levels, values, color=color, linewidth=2)
            ax.set_ylabel(ylabel)

        if self.is_rgb:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
            for ax, values, name in zip(axes, data, ["Red", "Green", "Blue"]):
                draw(ax, values, name.lower())
                ax.set_title(f"{name} channel - {title}")
            axes[-1].set_xlabel("Level")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            draw(ax, data, "gray" if kind == "bar" else "black")
            ax.set_xlabel("Level")
            ax.set_title(title)

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        plt.close(fig)  # important: avoids leaking figures between requests
        return buffer.getvalue()

    def graph(self):
        return self._plot(self._get_y(), "Histogram", "Frequency", "bar")

    def graph_pdf(self):
        return self._plot(self.pdf(), "PDF", "Probability", "bar")

    def graph_cdf(self):
        return self._plot(self.cdf(), "CDF", "Cumulative probability", "line")

    def graph_equalized(self):
        y = self._get_y()
        x_new = self.equalized_level(y)
        _, y_eq = self.equalize_histogram(y, x_new)
        return self._plot(y_eq, "Equalized histogram", "Frequency", "bar")