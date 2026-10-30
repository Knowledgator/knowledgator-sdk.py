import json
import logging
import pytest

logger = logging.getLogger(__name__)

REAL_DATASET_ID = "d655eda9-ca67-4a86-a8bc-5b3911233c79"


def _debug_print_samples(title: str, parsed: any, file_size: int):
    """Debug output helper to display dataset sample info (first, middle, last)."""
    print(f"\n{'='*25} DEBUG: {title} {'='*25}")
    print(f"File Size on Disk: {file_size} bytes")

    items = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        items = parsed["data"]

    total = len(items)
    print(f"Total Samples: {total}")

    if total == 0:
        print(f"Raw Object: {json.dumps(parsed, indent=2)[:500]}")
    elif total == 1:
        print(f"\n[Single Sample (0/1)]:\n{json.dumps(items[0], indent=2)}")
    elif total == 2:
        print(f"\n[First Sample (0/2)]:\n{json.dumps(items[0], indent=2)}")
        print(f"\n[Last Sample (1/2)]:\n{json.dumps(items[1], indent=2)}")
    else:
        mid_idx = total // 2
        print(f"\n[First Sample (0/{total})]:\n{json.dumps(items[0], indent=2)}")
        print(f"\n[Middle Sample ({mid_idx}/{total})]:\n{json.dumps(items[mid_idx], indent=2)}")
        print(f"\n[Last Sample ({total - 1}/{total})]:\n{json.dumps(items[-1], indent=2)}")
    print(f"{'='*65}\n")


def test_e2e_real_dataset_download_sync(sync_sdk_client, tmp_path):
    """Test downloading the real verified dataset using sync SDK client with stream and file_path."""
    dest_file = tmp_path / "real_dataset_download.json"

    # 1. Test download directly to file_path
    saved_path = sync_sdk_client.datasets.download(
        dataset_id=REAL_DATASET_ID,
        file_path=dest_file,
    )

    assert saved_path == dest_file
    assert dest_file.is_file()
    assert dest_file.stat().st_size > 0

    parsed = json.loads(dest_file.read_text(encoding="utf-8"))
    assert parsed is not None
    if isinstance(parsed, list):
        assert len(parsed) >= 1
    elif isinstance(parsed, dict):
        assert "data" in parsed or "text" in parsed or len(parsed) >= 1

    # Debug output: show first, middle, last samples for verification
    _debug_print_samples("Sync Download Result", parsed, dest_file.stat().st_size)

    # 2. Test stream_download yields chunks
    chunks = list(sync_sdk_client.datasets.stream_download(REAL_DATASET_ID, chunk_size=1024))
    assert len(chunks) >= 1
    assert b"".join(chunks) == dest_file.read_bytes()


@pytest.mark.asyncio
async def test_e2e_real_dataset_download_async(async_sdk_client, tmp_path):
    """Test downloading the real verified dataset using async SDK client with stream and file_path."""
    dest_file = tmp_path / "real_dataset_async.json"

    # 1. Test async download directly to file_path
    saved_path = await async_sdk_client.datasets.download_async(
        dataset_id=REAL_DATASET_ID,
        file_path=dest_file,
    )

    assert saved_path == dest_file
    assert dest_file.is_file()
    assert dest_file.stat().st_size > 0

    parsed_async = json.loads(dest_file.read_text(encoding="utf-8"))
    # Debug output: show first, middle, last samples for verification
    _debug_print_samples("Async Download Result", parsed_async, dest_file.stat().st_size)

    # 2. Test async stream_download_async
    async_chunks = []
    async for chunk in async_sdk_client.datasets.stream_download_async(REAL_DATASET_ID, chunk_size=1024):
        async_chunks.append(chunk)

    assert len(async_chunks) >= 1
    assert b"".join(async_chunks) == dest_file.read_bytes()


def test_e2e_real_dataset_upload_and_cleanup(sync_sdk_client, tmp_path):
    """Test uploading a dataset with exact name and cleaning it up."""
    exact_dataset_name = "e2e-real-upload-test"
    sample_data = [
        {
            "text": "This is the third time I've ordered this particular brand of coffee, and it never disappoints. The aroma when you open the bag is incredible, and the taste is consistently smooth with just the right amount of bitterness. I usually get the medium roast, and it's perfect for my morning pour-over. It's a bit pricier than some other brands, but the quality really shines through. Definitely my go-to for a quality cup of coffee.",
            "true_labels": [
                "positive"
            ],
            "all_labels": [
                "negative",
                "positive"
            ]
        }
    ]

    temp_file = tmp_path / "upload_sample.json"
    temp_file.write_text(json.dumps(sample_data, indent=2), encoding="utf-8")

    created_id = None
    try:
        res = sync_sdk_client.datasets.create(
            name=exact_dataset_name,
            type="comprehend-it",
            format="json",
            description="Live E2E upload verification using valid dataset format",
            file_path=temp_file,
        )

        assert res is not None
        # Extract created ID
        created_id = getattr(res, "id", None)
        if created_id is None and isinstance(res, dict):
            created_id = res.get("id")

        assert created_id is not None
        created_name = getattr(res, "name", None) or (res.get("name") if isinstance(res, dict) else None)

        if created_name:
            assert created_name == exact_dataset_name

        print(f"\n✅ Uploaded dataset ID: {created_id} (Name: '{created_name}')")

    finally:
        if created_id:
            try:
                sync_sdk_client.datasets.delete(dataset_id=created_id)
                logger.info("Successfully cleaned up test dataset: %s", created_id)
                print(f"✅ Cleaned up/deleted test dataset: {created_id}")
            except Exception as exc:
                logger.warning("Could not delete test dataset %s: %s", created_id, exc)

