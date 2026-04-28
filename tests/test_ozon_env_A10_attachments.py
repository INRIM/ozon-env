import base64
import os
import socket

import pytest
from ozonenv.OzonEnv import OzonEnv
from tests.test_common import auth_env
from pathlib import Path

pytestmark = pytest.mark.asyncio


def _is_service_up(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


async def test_db_insert_file_field_and_load_base64_dump():
    if not _is_service_up("localhost", 10002):
        pytest.skip("MongoDB test service is not available on localhost:10002")
    if not _is_service_up("localhost", 10765):
        pytest.skip(
            "Keycloak test service is not available on localhost:10765"
        )

    upload_dir = os.getenv("OZON_UPOLOAD_FOLDER", "/data/uploads")
    tmp_path = Path(os.environ.get("OZON_ENV_TMP_UPLOAD_FOLDER", "/tmp"))
    tmp_dir = tmp_path / "tmp"
    sample_file = tmp_path / "sample.txt"
    sample_file.write_bytes(b"db-attachment-content")

    env = None
    try:
        os.environ["OZON_ENV_TMP_UPLOAD_FOLDER"] = str(tmp_dir)
        env = OzonEnv()
        await auth_env(env)

        model = env.get("test_form_1")
        record = await model.new(
            {"rec_name": "attachment_1234", "firstName": "nume"}
        )
        record.addfile("uploadBase64", str(sample_file))
        saved = await model.insert(record)

        assert saved is not None
        assert len(saved.uploadBase64) == 1
        assert saved.uploadBase64[0]["filename"] == "sample.txt"

        stored_file = (
            Path(upload_dir) / "test_form_1" / "attachment_1234" / "sample.txt"
        )
        assert stored_file.read_bytes() == b"db-attachment-content"

        model.set_file_dump_mode("base64")
        loaded = await model.load({"rec_name": "attachment_1234"})

        assert loaded is not None
        assert len(loaded.uploadBase64) == 1
        assert (
            base64.b64decode(loaded.uploadBase64[0]["base64"])
            == b"db-attachment-content"
        )
    finally:
        if env:
            await env.close_env()
