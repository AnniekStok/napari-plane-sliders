try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._plane_slider_widget import PlaneSliderWidget

__all__ = ("PlaneSliderWidget",)
