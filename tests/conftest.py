"""Root conftest for all backend tests."""

from unittest.mock import MagicMock

import pytest

# ============================================================================
# Early Initialization - Runs before test collection
# ============================================================================


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    This is the earliest point where we can mock telemetry to prevent
    import-time errors when modules call get_tracer() and get_meter().
    """
    # Import telemetry using the SAME path that the source code uses
    # The source code uses: from src.backend.core.telemetry import ...
    from src.backend.core import telemetry

    # Create comprehensive mocks
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_span.is_recording = MagicMock(return_value=True)
    mock_span.set_attribute = MagicMock()
    mock_span.set_status = MagicMock()
    mock_span.record_exception = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span = MagicMock(return_value=mock_span)

    mock_counter = MagicMock()
    mock_histogram = MagicMock()

    mock_meter = MagicMock()
    mock_meter.create_counter = MagicMock(return_value=mock_counter)
    mock_meter.create_histogram = MagicMock(return_value=mock_histogram)

    # Directly set the module-level variables to prevent initialization checks
    telemetry._tracer = mock_tracer
    telemetry._meter = mock_meter


@pytest.fixture(autouse=True)
def reset_telemetry_mocks():
    """Reset telemetry mocks between tests to ensure isolation."""
    from src.backend.core import telemetry

    # Re-mock with fresh mocks for each test
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span = MagicMock(return_value=mock_span)

    mock_counter = MagicMock()
    mock_histogram = MagicMock()

    mock_meter = MagicMock()
    mock_meter.create_counter = MagicMock(return_value=mock_counter)
    mock_meter.create_histogram = MagicMock(return_value=mock_histogram)

    # Set fresh mocks
    telemetry._tracer = mock_tracer
    telemetry._meter = mock_meter

    yield

    # Keep mocks in place for next test (pytest_configure will reset on new session)
