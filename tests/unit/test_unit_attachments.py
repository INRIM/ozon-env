import base64
import os
from pathlib import Path

import pytest
from ozonenv.core.BaseModels import (
    BasicModel,
    Settings,
    User,
)
from ozonenv.core.ModelService import ServiceAttachment
from ozonenv.core.OzonOrm import OzonModel, OzonModelRest
from pydantic import Field

pytestmark = pytest.mark.asyncio


class AttachmentTestModel(BasicModel):
    doc: list[dict] = Field(default_factory=list)
    gallery: list[dict] = Field(default_factory=list)

    @classmethod
    def file_fields(cls) -> dict:
        return {
            "doc": {"multiple": False},
            "gallery": {"multiple": True},
        }


class DummyRestClient:
    def __init__(self):
        self.last_operation = ""
        self.last_payload = {}

    async def post_operation(self, operation_name: str, payload: dict):
        self.last_operation = operation_name
        self.last_payload = payload
        return {"data": payload["record"]}


class DummyEnv:
    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        self.db = None


class DummyOrm:
    def __init__(self, upload_folder: str):
        self.env = DummyEnv(upload_folder)
        self.app_settings = Settings(
            rec_name="test",
            upload_folder=upload_folder,
            tz="Europe/Rome",
        )
        self.rest_client = DummyRestClient()
        self.user_session = User(
            rec_name="adminuser",
            uid="adminuser",
            active=True,
        )
        self.private_models = []
        self.local_store = {}

    def is_local_model(self, _name: str) -> bool:
        return False


async def test_record_addfile_respects_single_and_multi_fields():
    record = AttachmentTestModel(rec_name="attachment.test")

    record.addfile("gallery", "/tmp/a.txt")
    record.addfile("gallery", "/tmp/b.txt")
    record.addfile("doc", "/tmp/doc-a.txt")

    assert record.gallery == ["/tmp/a.txt", "/tmp/b.txt"]
    assert record.doc == ["/tmp/doc-a.txt"]

    record.addfile("doc", "/tmp/doc-b.txt")

    assert record.doc == ["/tmp/doc-b.txt"]

    with pytest.raises(ValueError):
        record.addfile("doc", ["/tmp/1.txt", "/tmp/2.txt"])


async def test_service_attachment_save_files_dump_base64_and_transport(
    tmp_path,
):
    upload_dir = tmp_path / "uploads"
    tmp_dir = tmp_path / "tmp"
    service = ServiceAttachment()
    doc_path = tmp_path / "doc.txt"
    doc_path.write_bytes(b"doc-content")
    gallery_path = tmp_path / "gallery.txt"
    gallery_path.write_bytes(b"gallery-content")
    inline_payload = {
        "filename": "inline.txt",
        "content_type": "text/plain",
        "base64": base64.b64encode(b"inline-content").decode("utf-8"),
    }

    plan = await service.save_files(
        AttachmentTestModel,
        {
            "rec_name": "attachment.test",
            "doc": str(doc_path),
            "gallery": [str(gallery_path), inline_payload],
        },
        "attachment_model",
        "attachment.test",
    )

    assert len(plan.record["doc"]) == 1
    assert (
        plan.record["doc"][0]["file_path"]
        == "attachment_model/attachment.test"
    )
    assert len(plan.record["gallery"]) == 2
    assert len(plan.attachments_to_save) == 3

    for attachment in plan.attachments_to_save:
        await service.move_attachment(attachment)

    dumped = await service.dump_base64_files(
        AttachmentTestModel,
        plan.record,
    )
    assert base64.b64decode(dumped["doc"][0]["base64"]) == b"doc-content"
    dumped_gallery = {
        item["filename"]: base64.b64decode(item["base64"])
        for item in dumped["gallery"]
    }
    assert dumped_gallery["gallery.txt"] == b"gallery-content"
    assert dumped_gallery["inline.txt"] == b"inline-content"

    transport = await service.prepare_transport_files(
        AttachmentTestModel,
        {
            "doc": str(doc_path),
            "gallery": [str(gallery_path), plan.record["gallery"][1]],
        },
    )
    assert transport["doc"][0]["filename"] == "doc.txt"
    assert base64.b64decode(transport["doc"][0]["base64"]) == b"doc-content"
    assert transport["gallery"][1] == plan.record["gallery"][1]


async def test_rest_insert_transforms_file_paths_to_base64_payload(tmp_path):
    upload_dir = tmp_path / "uploads"
    orm = DummyOrm(str(upload_dir))
    model = OzonModelRest(
        "attachment_rest",
        orm,
        static=AttachmentTestModel,
    )
    await model.init_model()

    doc_path = tmp_path / "rest-doc.txt"
    doc_path.write_bytes(b"rest-content")
    record = await model.new({"rec_name": "attachment.rest"})
    record.addfile("doc", str(doc_path))

    saved = await model.insert(record)

    assert saved is not None
    assert orm.rest_client.last_operation == "insert"
    payload = orm.rest_client.last_payload["record"]
    assert payload["doc"][0]["filename"] == "rest-doc.txt"
    assert base64.b64decode(payload["doc"][0]["base64"]) == b"rest-content"


async def test_db_load_data_with_base64_dump_exposes_base64_on_record():

    upload_dir = Path(os.getenv("OZON_UPOLOAD_FOLDER", "./test/uploads"))
    attachment_dir = upload_dir / "attachment_model" / "attachment.record"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    file_path = attachment_dir / "sample.txt"
    file_path.write_bytes(b"db-load-content")

    orm = DummyOrm(str(upload_dir))
    model = OzonModel(
        "attachment_model",
        orm,
        static=AttachmentTestModel,
    )

    await model.init_model()
    model.set_file_dump_mode("base64")

    await model.load_data(
        {
            "rec_name": "attachment.record",
            "doc": [
                {
                    "filename": "sample.txt",
                    "content_type": "text/plain",
                    "file_path": "attachment_model/attachment.record",
                    "url": "/attachment_model/attachment.record/sample.txt",
                    "key": "attachment.record",
                }
            ],
        }
    )

    assert model.modelr.doc[0]["filename"] == "sample.txt"
    assert (
        base64.b64decode(model.modelr.doc[0]["base64"]) == b"db-load-content"
    )
