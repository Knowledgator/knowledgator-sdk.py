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

    def test_start_training_adapter_payload(self, api):
        """Test start_training with adapter payload structure matching working JSON spec."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ft-adapter-1",
            "model_id": "comprehend-it",
            "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
            "status": "pending",
            "created_at": "2026-09-16T00:00:00Z",
            "updated_at": "2026-09-16T00:00:00Z",
        }
        api.configs.client.request.return_value = mock_response

        res = api.start_training(
            model_id="comprehend-it",
            dataset_id="6babd48a-2256-4466-b16b-9fd45ba59688",
            type="adapter",
            hyperparameters={
                "seed": 0,
                "batch": None,
                "learning_rate_mult": None,
                "num_of_epochs": None,
            },
            validation_dataset_id="6babd48a-2256-4466-b16b-9fd45ba59688",
        )

        assert res.id == "ft-adapter-1"
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["json"] == {
            "parent_model_id": "comprehend-it",
            "type": "adapter",
            "training": {
                "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
                "args": {
                    "seed": 0,
                    "batch": None,
                    "learning_rate_mult": None,
                    "num_of_epochs": None,
                },
            },
            "validation": {
                "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
            },
        }

    def test_start_training_full_model_payload(self, api):
        """Test start_training with full model payload structure matching working JSON spec."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ft-model-1",
            "model_id": "gliclass-x-base",
            "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
            "status": "pending",
            "created_at": "2026-09-16T00:00:00Z",
            "updated_at": "2026-09-16T00:00:00Z",
        }
        api.configs.client.request.return_value = mock_response

        res = api.start_training(
            model_id="gliclass-x-base",
            dataset_id="6babd48a-2256-4466-b16b-9fd45ba59688",
            type="model",
            hyperparameters={
                "seed": 0,
                "batch": None,
                "learning_rate_mult": None,
                "num_of_epochs": None,
            },
            validation_dataset_id="6babd48a-2256-4466-b16b-9fd45ba59688",
        )

        assert res.id == "ft-model-1"
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["json"] == {
            "parent_model_id": "gliclass-x-base",
            "type": "model",
            "training": {
                "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
                "args": {
                    "seed": 0,
                    "batch": None,
                    "learning_rate_mult": None,
                    "num_of_epochs": None,
                },
            },
            "validation": {
                "dataset_id": "6babd48a-2256-4466-b16b-9fd45ba59688",
            },
        }

    def test_list_models_with_filters(self, api):
        """Test listing models with full model and adapter filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "data": [
                {
                    "id": "gliclass-v4-2b",
                    "name": "gliclass-v4-2b",
                    "type": "GLiClass",
                    "fine_tuned": False,
                    "inference": "open-models",
                    "training": {"model": False, "adapter": False},
                }
            ],
            "offset": 0,
            "limit": 20,
        }
        api.configs.client.request.return_value = mock_response

        res = api.list_models(
            offset=0,
            limit=20,
            object_type="model",
            trainable="model",
            inference="open-models",
            type="GLiClass",
            name="gliclass",
        )

        assert res.total == 1
        assert len(res.models) == 1
        assert res.models[0].id == "gliclass-v4-2b"
        assert res.models[0].type == "GLiClass"
        assert res.models[0].inference == "open-models"

        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"].path == "/models/"
        assert str(kwargs["url"]) == "https://api.knowledgator.com/models/?offset=0&limit=20&object_type=model&trainable=model&inference=open-models&type=GLiClass&name=gliclass"

    @pytest.mark.asyncio
    async def test_list_models_async_with_filters(self, api):
        """Test async listing models with adapter filters."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "data": [
                {
                    "id": "adapter-1",
                    "name": "adapter-1",
                    "type": "GLiClass",
                    "fine_tuned": True,
                    "parent_model": "gliclass-v4-2b",
                    "inference": "comprehend-it",
                    "training": {"model": False, "adapter": True},
                }
            ],
            "offset": 0,
            "limit": 10,
        }
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        res = await api.list_models_async(
            offset=0,
            limit=10,
            object_type="adapter",
            trainable="adapter",
            inference="comprehend-it",
        )

        assert res.total == 1
        assert len(res.models) == 1
        assert res.models[0].id == "adapter-1"
        assert res.models[0].fine_tuned is True
        assert res.models[0].parent_model == "gliclass-v4-2b"

        _args, kwargs = api.configs.async_client.request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"].path == "/models/"
        assert str(kwargs["url"]) == "https://api.knowledgator.com/models/?offset=0&limit=10&object_type=adapter&trainable=adapter&inference=comprehend-it"

    def test_get_model_endpoint(self, api):
        """Test get_model calls /models/{id} as public endpoint with no auth header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "model-abc",
            "name": "model-abc",
            "description": "Full model description markdown",
            "type": "GLiClass",
        }
        api.configs.client.request.return_value = mock_response

        res = api.get_model("model-abc")

        assert res.id == "model-abc"
        assert res.name == "model-abc"
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"].path == "/models/model-abc/"
        assert "Authorization" not in kwargs["headers"]
        assert "KnowledgatorAPI-Key" not in kwargs["headers"]

    @pytest.mark.asyncio
    async def test_get_model_async_endpoint(self, api):
        """Test async get_model_async calls /models/{id} as public endpoint with no auth header."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "model-xyz",
            "name": "model-xyz",
            "description": "Full markdown",
            "type": "GLiClass",
        }
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        res = await api.get_model_async("model-xyz")

        assert res.id == "model-xyz"
        _args, kwargs = api.configs.async_client.request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"].path == "/models/model-xyz/"
        assert "Authorization" not in kwargs["headers"]
        assert "KnowledgatorAPI-Key" not in kwargs["headers"]

    def test_list_trainings_with_type(self, api):
        """Test listing trainings with type query parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "trainings": [
                {"id": "tr-1", "type": "comprehend-it", "status": "completed"}
            ],
            "total": 1,
            "offset": 0,
            "limit": 20,
        }
        api.configs.client.request.return_value = mock_response

        res = api.list_trainings(status="completed", type="comprehend-it", limit=20)

        assert res.total == 1
        assert len(res.trainings) == 1
        _args, kwargs = api.configs.client.request.call_args
        assert kwargs["method"] == "GET"
        assert "/trainings" in str(kwargs["url"])
        assert "type=comprehend-it" in str(kwargs["url"])
        assert "status=completed" in str(kwargs["url"])

    @pytest.mark.asyncio
    async def test_list_trainings_async_with_type(self, api):
        """Test async listing trainings with type query parameter."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "trainings": [
                {"id": "tr-2", "type": "extract-it", "status": "completed"}
            ],
            "total": 1,
            "offset": 0,
            "limit": 10,
        }
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        res = await api.list_trainings_async(status="completed", type="extract-it", limit=10)

        assert res.total == 1
        assert len(res.trainings) == 1
        _args, kwargs = api.configs.async_client.request.call_args
        assert kwargs["method"] == "GET"
        assert "/trainings" in str(kwargs["url"])
        assert "type=extract-it" in str(kwargs["url"])
        assert "status=completed" in str(kwargs["url"])
