"""Unit tests for src/util/logger.py."""

from unittest.mock import patch

import pytest

from src.util.logger import Logger


@pytest.mark.parametrize(
    ("method_name", "expected_call"),
    [
        ("debug", ("Debug %s", "value")),
        ("info", ("Info %s", "value")),
        ("warning", ("Warning %s", "value")),
        ("error", ("Error %s", "value")),
    ],
)
def test_logger_methods_support_lazy_formatting(method_name, expected_call):
    logger = Logger()
    internal_logger = logger._Logger__logger

    with patch.object(internal_logger, method_name) as mocked_method:
        getattr(logger, method_name)(*expected_call)

    mocked_method.assert_called_once_with(*expected_call)


def test_logger_methods_forward_keyword_arguments():
    logger = Logger()
    internal_logger = logger._Logger__logger

    with patch.object(internal_logger, "error") as mocked_error:
        logger.error("Failure %s", "X", exc_info=True)

    mocked_error.assert_called_once_with("Failure %s", "X", exc_info=True)
