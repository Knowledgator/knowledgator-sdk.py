"""
Mock API Request Tests for Finetuning Endpoint
"""

from unittest.mock import MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.apis.finetuning import FinetuningAPI
from knowledgator.finetuning.common import TrainingStatus


class TestFinetuningAPIMockRequests:
    """Test Finetuning API with mock HTTP requests."""

    @pytest.fixture
    def api(self):
        """Create FinetuningAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return FinetuningAPI(configs)

    def test_finetuning_defaults(self, api):
        """Test default values for FinetuningAPI."""
        assert api.PATH == "/trainings/"
        assert api.METHOD == "GET"  # finetuning.py says METHOD = "GET"
        assert api.CONTENT_TYPE == "application/json"

    def test_list_trainings_status_check(self, api):
        """Test listing trainings and verifying status enums."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "trainings": [
                {
                    "id": "ft-1",  # Changed from job_id
                    "model_id": "model-1",
                    "dataset_id": "dataset-1",
                    "status": "active",
                    "created_at": "2024-01-01",
                    "updated_at": "2024-01-01",
                },
                {
                    "id": "ft-2",
                    "model_id": "model-2",
                    "dataset_id": "dataset-2",
                    "status": "completed",
                    "created_at": "2024-01-01",
                    "updated_at": "2024-01-01",
                },
            ],
            "total": 2,
            "offset": 0,
            "limit": 100,
        }
        api.configs.client.request.return_value = mock_response

        result = api.list_trainings()

        assert len(result.trainings) == 2

        # Verify status is correctly parsed/returned
        assert result.trainings[0].status == TrainingStatus.ACTIVE
        assert result.trainings[1].status == TrainingStatus.COMPLETED

        # Verify request details
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "GET"
        assert "/trainings" in str(kwargs["url"])

    def test_get_training_status(self, api):
        """Test getting a single training and verifying status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ft-3",
            "model_id": "model-3",
            "dataset_id": "dataset-3",
            "status": "failed",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "error": "OOM",
        }
        api.configs.client.request.return_value = mock_response

        result = api.get_training("ft-3")

        assert result.id == "ft-3"
        assert result.status == TrainingStatus.FAILED

        # Verify request details
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "GET"
        # Base path /trainings/ + {id}/
        assert "/trainings/ft-3" in str(kwargs["url"])

    def test_stop_training(self, api):
        """Test stopping a training."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # stop_training uses POST and expects empty body response or just 200 OK
        # The return type is None
        mock_response.json.return_value = {}
        api.configs.client.request.return_value = mock_response

        api.stop_training("ft-4")

        assert api.configs.client.request.called

        # Verify request details
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "POST"

        # Verify URL construction
        # Base path /trainings/ + {id}/stop/
        assert "/trainings/ft-4/stop" in str(kwargs["url"])

        # Verify empty json body
        assert kwargs["json"] == {}
