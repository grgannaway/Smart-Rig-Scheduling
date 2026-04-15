from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def test_project_main(
    caplog: "LogCaptureFixture",
) -> None:
    from project.__main__ import main

    main()
    assert caplog.text is not None


def test_project_version() -> None:
    from project import __version__

    major, minor, patch = map(int, (__version__.split(".")))
    assert major >= 0
    assert minor >= 0
    assert patch >= 0


@pytest.mark.parametrize(
    "major_increment",
    [
        pytest.param(0, id="0"),
        pytest.param(1, id="1"),
    ],
)
def test_deprecated(major_increment: int) -> None:
    import warnings
    from typing import overload

    from ml_utils import overload_dispatch
    from typing_extensions import deprecated

    from project import __version__ as version

    major, minor, patch = map(int, (version.split(".")))

    deprecation_message = (
        f">{major - major_increment}.{minor}.{patch}, use instead: new_function"
    )

    if major_increment > 0:
        # any deprecated functions older than the current major version raise exception
        warnings.filterwarnings(
            "error",
            message=rf"^>{major-1}\.",
            category=DeprecationWarning,
        )

    @overload
    @deprecated(deprecation_message)
    def deprecated_function(value: str) -> None:
        pass

    @overload_dispatch
    def deprecated_function(value: int) -> None:
        pass

    @deprecated(deprecation_message)
    class DeprecatedClass:
        pass

    class DeprecatedMethod:
        @deprecated(deprecation_message)  # pyright: ignore[reportArgumentType]
        def deprecated_method(self) -> None:
            pass

    # test that the deprecation warning is not raised for int input type
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deprecated_function(value=1)

    # test warning if major_increment is 0, otherwise test exception
    context = pytest.warns if major_increment == 0 else pytest.raises

    with context(DeprecationWarning) as record:
        deprecated_function(value="test")
        assert str(record[0].message) == deprecation_message
    with context(DeprecationWarning) as record:
        DeprecatedClass()
        assert str(record[0].message) == deprecation_message
    with context(DeprecationWarning) as record:
        DeprecatedMethod().deprecated_method()
        assert str(record[0].message) == deprecation_message


if __name__ == "__main__":
    import sys

    sys.path.append("project")
    pytest.main()
