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
* `Link-it` (coming soon): Zero-shot entity-linking tool that supports entity extraction, disambiguation, and linking to custom databases.
* `Instruct-it` (coming soon): Extraction of structured information from raw text based on provided schema.

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

Before using the SDK, obtain a Knowledgator API key from [api.knowledgator.com](https://api.knowledgator.com/account/api-keys).

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

response = client.comprehend.it("Earth is beautiful", labels=["positive", "negative", "neutral"])

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
    labels=["geography", "sport"],
    threshold=0.5,
    examples=[{"text": "Example text", "label": "geography"}]
)

print(res)  # Outputs ComprehendResults with classifications
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
        labels=["geography", "sport"],
        threshold=0.5,
        examples=[{"text": "Example text", "label": "geography"}]
    )

    print(res)

asyncio.run(main())
```

## Table of Contents

- [Information Extraction (Extract)](#information-extraction-extract)
- [Relation Extraction](#relation-extraction)
- [Summarization](#summarization)
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
    labels=["person", "location"],
    threshold=0.5,
    prompt="Custom prompt for extraction"
)

print(res)  # Outputs ExtractBaseResults with entities
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
        labels=["person", "location"],
        threshold=0.5,
        prompt="Custom prompt for extraction"
    )

    print(res)

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
- `parse_labels`: Parse raw response into ComprehendResults.

### ExtractAPI

- `it`: Zero-shot NER.
- `it_async`: Asynchronous NER.
- `relations`: Relation extraction.
- `relations_async`: Asynchronous relation extraction.
- `summary`: Text summarization.
- `summary_async`: Asynchronous summarization.
- `_parse_entities`: Parse entities into ExtractBaseResults.
- `_parse_relations`: Parse relations into ExtractRelationsResults.
- `_parse_summary`: Parse summary into ExtractSummaryResults.

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
    res = kg.comprehend.it("text", ["labels"])
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