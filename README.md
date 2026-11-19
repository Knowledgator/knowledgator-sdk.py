# Knowledgator Python SDK

[![version badge](https://img.shields.io/pypi/v/knowledgator)](https://pypi.org/project/knowledgator/)
![license badge](https://img.shields.io/github/license/knowledgator-ai/knowledgator-python)

The Knowledgator Python SDK provides access to the Knowledgator APIs for zero-shot classification and information extraction tasks. It supports both synchronous and asynchronous operations, with variants for standard API access and RapidAPI integration.

### Key Advantages
* **Efficiency**: Based on non-generative technologies, our solutions are *100×* faster and more cost-efficient.
* **Precision**: Designed and tailored to information extraction needs, our models demonstrate better precision than LLMs.
* **Interpretability**: Every processing result comes in structured form with a confidence score, offering a reliable indicator of precision to guide your trust in automation.
* **Customisability**: Our models can be fine-tuned with a few examples to achieve results on par with supervised models.
* **Modularity**: Naturally providing structured outputs, our APIs can be chained to solve complex end-to-end information extraction tasks.

### Available Models
* `Comprehend-it`: An efficient and accurate zero-shot and few-shot classifier ideal for text.
* `Extract-it`: Fast, interpretable, and accurate information extraction tool that efficiently handles different extraction tasks, like NER and relation extraction.
* `Instruct-it`: Extraction of structured information from raw text based on provided schema.
* `Link-it` (coming soon): Zero-shot entity-linking tool that supports entity extraction, disambiguation, and linking to custom databases.

### Available Services
* `Datasets`: Manage datasets and raw data for NER, text classification, and relation extraction.
* `Fine-tuning`: Fine-tune models with your own datasets for improved performance.
* `Open Models`: Access finetuned or custom models using dynamic identifiers.

## Documentation

Knowledgator API reference and guides are available [here](https://docs.knowledgator.com/). For detailed endpoint specifications, refer to the API documentation.

## Quick Installation

> **Python Version Policy**: Supports Python 3.8+. Versions reaching end-of-life get a 3-month grace period before updates.

Install via pip:

```bash
pip install knowledgator
```

Or with Poetry:

```bash
poetry add knowledgator
```

For shell/script usage with uv:

```shell
uvx --from knowledgator python
```

You can also install from source by cloning the repo:

```bash
git clone git@github.com:Knowledgator/knowledgator-sdk.py.git
cd knowledgator-sdk.py
pip install -e .
```

## Quick Start

### API Key Setup

Before using the SDK, obtain a Knowledgator API key from [platform.knowledgator.com](https://platform.knowledgator.com/account/api-keys).

Set your API key as an environment variable:

```bash
$ export KG_API_KEY="your_api_key_here"
```

You can write it to zsh in the following way:

```bash
# Set Knowledgator API Key (using zsh for example)
$ echo 'export KG_API_KEY=[your_key_here]' >> ~/.zshenv

# Reload the environment (or open a new terminal)
$ source ~/.zshenv
```

Alternatively, pass the key directly when initializing the client.

### SDK Example Usage

```python
from knowledgator import Knowledgator

client = Knowledgator(api_key="YOUR API KEY")

response = client.comprehend.it(
    text="Earth is beautiful",
    tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}}
)

print(response)
```

### Basic Classification (Comprehend)

Use the `comprehend.it` method for zero-shot text classification.

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.comprehend.it(
    text="Earth is beautiful",
    tasks={
        "topic": {
            "type": "classify",
            "labels": ["geography", "sport"],
            "threshold": 0.5,
            "examples": [{"text": "Example text", "label": "geography"}],
        }
    },
)

print(res)  # Outputs ComprehendResults with classifications
print(res.topic)  # Directly access parsed task output
```

For asynchronous usage:

```python
# Asynchronous Example
import asyncio
from knowledgator import Knowledgator
import os

async def main():
    kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

    res = await kg.comprehend.it_async(
        text="Kyiv is beautiful",
        tasks={
            "topic": {
                "type": "classify",
                "labels": ["geography", "sport"],
                "threshold": 0.5,
                "examples": [{"text": "Example text", "label": "geography"}],
            }
        },
    )

    print(res)
    print(res.topic)

asyncio.run(main())
```

## Table of Contents

- [Information Extraction (Extract)](#information-extraction-extract)
- [Relation Extraction](#relation-extraction)
- [Summarization](#summarization)
- [Structured Extraction (Instruct)](#structured-extraction-instruct)
- [Dataset Management](#dataset-management)
- [Raw Data Management](#raw-data-management)
- [Fine-tuning](#fine-tuning)
- [Open Models](#open-models)
- [Model Types: LoRA Adapters vs. Open Models](#model-types-lora-adapters-vs-open-models)
- [RapidAPI Variant](#rapidapi-variant)
- [Available Resources and Operations](#available-resources-and-operations)
- [Retries](#retries)
- [Error Handling](#error-handling)
- [Custom HTTP Client](#custom-http-client)
- [Authentication](#authentication)
- [Resource Management](#resource-management)
- [Debugging](#debugging)
- [Development](#development)

## Information Extraction (Extract)

Use the `extract.it` method for zero-shot named entity recognition (NER).

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.extract.it(
    text="Extract entities from this text.",
    tasks={
        "entities": {
            "type": "span",
            "labels": ["person", "location"],
            "threshold": 0.5,
            "instruction": "Custom prompt for extraction",
        }
    },
)

print(res)  # Outputs ExtractResults with entities
print(res.entities)  # Directly access parsed entities task output
```

Asynchronous version:

```python
# Asynchronous Example
import asyncio
from knowledgator import Knowledgator
import os

async def main():
    kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

    res = await kg.extract.it_async(
        text="Extract entities from this text.",
        tasks={
            "entities": {
                "type": "span",
                "labels": ["person", "location"],
                "threshold": 0.5,
                "instruction": "Custom prompt for extraction",
            }
        },
    )

    print(res)
    print(res.entities)

asyncio.run(main())
```

## Relation Extraction

Use `extract.relations` for extracting relations between entities.

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.extract.relations(
    text="Text with entities and relations.",
    entities=["person", "organization"],
    relations=["works_for", "located_in"],
    threshold=0.5,
    prompt="Custom relation prompt",
    rules=[{"rule": "example rule"}]
)

print(res)  # Outputs ExtractRelationsResults
```

Asynchronous: Use `relations_async`.

## Summarization

Use `extract.summary` for text summarization.

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.extract.summary(
    text="Long text to summarize.",
    threshold=0.5,
    prompt="Custom summarization prompt"
)

print(res)  # Outputs ExtractSummaryResults
```

Asynchronous: Use `summary_async`.

## Structured Extraction (Instruct)

Use `instruct.it` for extracting structured information based on a schema.

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.instruct.it(
    text="John Doe works at Acme Corp as a Software Engineer.",
    template='{"name": "", "company": "", "position": ""}',
    output_type="json"
)

print(res)  # Outputs InstructResults with structured data
```

Asynchronous version:

```python
# Asynchronous Example
import asyncio
from knowledgator import Knowledgator
import os

async def main():
    kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

    res = await kg.instruct.it_async(
        text="John Doe works at Acme Corp as a Software Engineer.",
        template='{"name": "", "company": "", "position": ""}',
        output_type="json"
    )

    print(res)

asyncio.run(main())
```

## Dataset Management

Manage datasets for training and fine-tuning models.

### List Datasets

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.datasets.list(
    offset=0,
    limit=10,
    type="ner"
)

print(res)  # Outputs ListDatasetsResponse
```

### Create Dataset

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.datasets.create(
    name="My NER Dataset",
    type="ner",
    format="jsonl",
    short_description="Dataset for named entity recognition"
)

print(res)  # Outputs DatasetCreationResponse
```

### Upload File to Dataset

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

with open("data.jsonl", "rb") as f:
    res = kg.datasets.upload(dataset_id="dataset-id", file=f)

print(res)  # Outputs DatasetCreationResponse
```

## Raw Data Management

Raw data entries can be managed similarly to datasets using the `datasets` API.

### List Raw Data

```python
res = kg.datasets.list_data(limit=10)
print(res)  # Outputs ListDataResponse
```

### Create Raw Data

```python
res = kg.datasets.create_data(
    name="My Raw Data",
    type="ner",
    format="jsonl",
    file=open("raw_data.jsonl", "rb")
)
print(res)
```

## Fine-tuning

Fine-tune models with your own datasets.

### Start Training

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.finetuning.start_training(
    model_id="base-model-id",
    dataset_id="dataset-id",
    hyperparameters={
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001
    }
)

print(res)  # Outputs TrainingObject
```

### List Training Jobs

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.finetuning.list_trainings(
    offset=0,
    limit=10,
    status="running"
)

print(res)  # Outputs ListTrainingsResponse
```

### Get Training Metrics

```python
# Synchronous Example
from knowledgator import Knowledgator
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

res = kg.finetuning.get_metrics(
    training_id="training-id",
    metric="loss",
    offset=0,
    limit=100
)

print(res)  # Outputs ListMetricsResponse
```

## Open Models

Interact with custom or finetuned models using dynamic model identifiers.

```python
# Initialize a model client for a standalone full model
model = kg.model("m-d9be7ede307943439a30c6f814026566")

# Default prediction invokes POST /inference/{model_id}/
response = model.predict({
    "text": "Smartphone with 120Hz OLED screen and 5000mAh battery",
    "labels": ["Electronics", "Apparel", "Home Goods"]
})

# Shorthand for method-based inference (invokes POST /inference/{model_id}/{method}/)
extractor = kg.model("m-d9be7ede307943439a30c6f814026566", method="relations")
response = extractor.predict({"text": "Apple released iPhone in Cupertino."})

# Explicitly call a specific sub-method
result = model.call("relations", {"text": "Apple released iPhone in Cupertino."})
```

## Model Types: LoRA Adapters vs. Open Models

Knowledgator supports two distinct types of fine-tuned and custom models:

| Model Type | Platform Representation | Where to Run Inference | How to Call |
| :--- | :--- | :--- | :--- |
| **LoRA Adapters** | `type="adapter"` (attached to a base service like `comprehend-it` or `extract-it`) | Base Service Endpoint (`/inference/comprehend/`, `/inference/extract/`) with `KnowledgatorAPI-Model-ID` header | `kg.comprehend.it(..., adapter_id="<adapter-id>")` or `kg.extract.it(..., adapter_id="<adapter-id>")` |
| **Open Models** | `type="model"` (standalone full model, e.g. fine-tuned `gliclass-x-base`, `gliner-base`) | Dedicated Endpoint (`/inference/{model_id}/` or `/inference/{model_id}/{method}/`) | `kg.model("<model-id>").predict(...)` or `kg.model("<model-id>", method="...").predict(...)` |

### 1. Using a LoRA Adapter

When fine-tuning creates a lightweight adapter (`type="adapter"`), supply `adapter_id`:

```python
# Zero-shot classification with your fine-tuned LoRA adapter
res = kg.comprehend.it(
    text="Smartphone with 120Hz OLED screen and 5000mAh battery",
    tasks={"topic": {"type": "classify", "labels": ["Electronics", "Apparel", "Home Goods"]}},
    adapter_id="your-adapter-id"  # Automatically passes KnowledgatorAPI-Model-ID header
)
```

### 2. Using a Full Open Model

When fine-tuning produces a standalone model or when using standalone full models (`type="model"`), route through `kg.model(...)`:

```python
# Predict using a standalone full open model (e.g. fine-tuned gliclass or gliner)
model = kg.model("m-d9be7ede307943439a30c6f814026566")

# The input payload matches the expected schema of the model (e.g. text + labels)
res = model.predict({
    "text": "Smartphone with 120Hz OLED screen and 5000mAh battery",
    "labels": ["Electronics", "Apparel", "Home Goods"]
})
print(res)
```

> ⚠️ **Important**: Do not pass a standalone Open Model ID as `adapter_id` to `comprehend.it` or `extract.it`. The base service will reject it with HTTP 503 (`server.resource_unavailable`). Always use `kg.model(model_id).predict(...)` for Open Models.


## RapidAPI Variant

For RapidAPI-hosted endpoints:

```python
from knowledgator import KnowledgatorRapidAPI
import os

kg_rapid = KnowledgatorRapidAPI(api_key=os.getenv("RAPID_KG_API_KEY", ""))

# Usage similar to above, e.g., kg_rapid.comprehend.it(...)
```

## Available Resources and Operations

### ComprehendAPI

- `it`: Zero-shot classification.
- `it_async`: Asynchronous classification.

### ExtractAPI

- `it`: Zero-shot NER.
- `it_async`: Asynchronous NER.
- `relations`: Relation extraction.
- `relations_async`: Asynchronous relation extraction.
- `summary`: Text summarization.
- `summary_async`: Asynchronous summarization.

### InstructAPI

- `it`: Structured information extraction.
- `it_async`: Asynchronous structured information extraction.

### DatasetsAPI

- `list`: List datasets with optional filtering.
- `list_async`: Asynchronous list datasets.
- `create`: Create a new dataset.
- `create_async`: Asynchronous create dataset.
- `get`: Get dataset details by ID.
- `get_async`: Asynchronous get dataset.
- `update`: Update dataset metadata.
- `update_async`: Asynchronous update dataset.
- `delete`: Delete a dataset.
- `delete_async`: Asynchronous delete dataset.
- `upload`: Upload a file to an existing dataset.
- `upload_async`: Asynchronous upload file.
- `list_formats`: List available dataset formats.
- `list_formats_async`: Asynchronous list formats.

### FinetuningAPI

- `create_base_model`: Create a new base model for training.
- `create_base_model_async`: Asynchronous create base model.
- `list_models`: List available base models.
- `list_models_async`: Asynchronous list models.
- `get_model`: Get model details by ID.
- `get_model_async`: Asynchronous get model.
- `list_model_types`: List available model types.
- `list_model_types_async`: Asynchronous list model types.
- `start_training`: Start a new training job.
- `start_training_async`: Asynchronous start training.
- `list_trainings`: List training jobs with optional filtering.
- `list_trainings_async`: Asynchronous list trainings.
- `get_training`: Get training job details by ID.
- `get_training_async`: Asynchronous get training.
- `delete_training`: Delete a training job.
- `delete_training_async`: Asynchronous delete training.
- `stop_training`: Stop a running training job.
- `stop_training_async`: Asynchronous stop training.
- `list_events`: List events for a training job.
- `list_events_async`: Asynchronous list events.
- `get_metrics`: Get training metrics for a specific metric type.
- `get_metrics_async`: Asynchronous get metrics.

### Configurations

- `SDKConfigs`: Manages clients, servers, retries, timeouts.
- `RetryPolicyConfigs`: Configures retry logic (max_tries, backoff, jitter).
- `ServersConfigs`: Handles server URLs and headers.

### HTTP Clients

- `HttpClient`: Synchronous HTTP requests with retries.
- `AsyncHttpClient`: Asynchronous HTTP requests with retries.

## Retries

Operations support retries via `RetryPolicyConfigs`. Default: 3 tries, exponential backoff with jitter.

Override per-operation or globally:

```python
from knowledgator import Knowledgator, RetryPolicyConfigs
import os

kg = Knowledgator(
    api_key=os.getenv("KG_API_KEY", ""),
    retry_config=RetryPolicyConfigs(max_tries=5, base_delay=0.5)
)
```

## Error Handling

Errors raise exceptions like `ServerError`, `TransportTimeout`, `ConnectionFailure`. Catch and handle:

```python
from knowledgator import Knowledgator
from knowledgator.exceptions import ServerError
import os

kg = Knowledgator(api_key=os.getenv("KG_API_KEY", ""))

try:
    res = kg.comprehend.it(
        text="text",
        tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}}
    )
except ServerError as e:
    print(e.status_code, e.payload)
except Exception as e:
    raise e
```

## Custom HTTP Client

Use a custom `httpx.Client` or `httpx.AsyncClient`:

```python
import httpx
from knowledgator import Knowledgator

custom_client = httpx.Client(headers={"Custom-Header": "value"})
kg = Knowledgator(api_key="sk-...", client=custom_client)
```

## Authentication

Authenticate via `api_key` parameter or `KG_API_KEY` env var. For RapidAPI, use `RAPID_KG_API_KEY`.

## Resource Management

`Knowledgator` supports context managers for closing clients:

```python
with Knowledgator(api_key="sk-...") as kg:
    # Usage here
```

Async: Use `async with`.

## Debugging

Enable debug logging:

```python
import logging
from knowledgator import Knowledgator

logging.basicConfig(level=logging.DEBUG)
kg = Knowledgator(api_key="sk-...", debug_logger=logging.getLogger("knowledgator"))
```

Or set `KNOWLEDGATOR_DEBUG=true` env var.

## Contributing
Feel free to contribute and open issues or PRs.

1. Fork → feature branch → PR.
2. For docs tweaks (like this README) – patch away, no generator sync needed.
3. For runtime changes – supply tests; we’ll migrate logic upstream.
