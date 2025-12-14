"""
Unit tests for configuration management.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.utils.config_manager import get_device


class TestDeviceManagement:
    """Test GPU device detection and management."""

    def test_get_device_logic(self):
        """Test device selection logic."""
        # Test that get_device returns a valid device
        device = get_device()
        assert device in ["cuda", "cpu"]

    @patch('torch.cuda.is_available')
    def test_get_device_cuda_disabled(self, mock_cuda):
        """Test device selection when CUDA is disabled in config."""
        mock_cuda.return_value = True

        with patch('src.utils.config_manager.get_settings') as mock_settings:
            mock_settings.return_value.general.gpu_enabled = False
            device = get_device()
            assert device == "cpu"

    @patch('torch.cuda.is_available')
    def test_get_device_no_cuda(self, mock_cuda):
        """Test device selection when CUDA is not available."""
        mock_cuda.return_value = False

        device = get_device()
        assert device == "cpu"
