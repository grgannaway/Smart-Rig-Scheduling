"""The project package."""

import logging
import warnings as _warnings

from ._version import __version__

_major, _minor, _patch = map(int, (__version__.split(".")))

# any deprecated functions older than the current major version will raise exception
_warnings.filterwarnings(
    "error",
    message=rf"^>{_major-1}\.",
    category=DeprecationWarning,
)


package_logger = logging.getLogger(__name__)

package_logger.setLevel(logging.INFO)
