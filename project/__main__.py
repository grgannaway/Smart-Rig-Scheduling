import logging
import sys

from . import __version__

module_logger = logging.getLogger(__name__)


def main() -> None:
    """The main entry point of the application."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    module_logger.info(
        "Running %s version %s",
        __package__,
        __version__,
    )


if __name__ == "__main__":
    main()
