import asyncio
import base64
import binascii
import copy
import logging
import mimetypes
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import aiofiles
from pydantic import AwareDatetime
from starlette.responses import FileResponse

from ozonenv.core.BaseModels import CoreModel, Settings, OzonEnvCoreSettings
from ozonenv.core.DataValueService import DataValueService
from ozonenv.core.DateEngine import DateEngine

# from ozonenv.core.OzonOrm import OzonOrm
logger = logging.getLogger(__name__)


class AttachmentError(ValueError):
    pass


@dataclass(frozen=True)
class AttachmentSavePlan:
    record: dict
    attachments_to_save: list[dict]


@dataclass
class Base64Upload:
    filename: str
    content_type: str
    content: bytes
    offset: int = 0

    async def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.content):
            return b""
        if size is None or size < 0:
            size = len(self.content) - self.offset
        start = self.offset
        self.offset = min(len(self.content), self.offset + size)
        return self.content[start : self.offset]


def _base64_upload_from_dict(value: Any) -> Base64Upload | None:
    if not isinstance(value, dict):
        return None
    raw_data = _raw_base64_content(value)
    if not raw_data:
        return None
    filename = (
        value.get("filename")
        or value.get("name")
        or value.get("originalName")
        or value.get("original_name")
        or ""
    )
    if not filename:
        raise AttachmentError("Missing base64 attachment filename")
    content_type = (
        value.get("content_type")
        or value.get("type")
        or value.get("mimetype")
        or "application/octet-stream"
    )
    encoded, data_url_content_type = _strip_data_url(str(raw_data))
    if data_url_content_type:
        content_type = data_url_content_type
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError(
            f"Invalid base64 attachment content: {filename}"
        ) from exc
    return Base64Upload(
        filename=str(filename),
        content_type=str(content_type),
        content=content,
    )


def _raw_base64_content(value: dict) -> Any:
    for key in ("content", "base64", "data"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    url = value.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        return url
    return None


def _resolve_inside(root: Path, relative_path: Path) -> Path:
    root = root.expanduser().resolve(strict=False)
    target = (root / relative_path).resolve(strict=False)
    if target != root and root not in target.parents:
        raise AttachmentError(f"Invalid attachment path: {relative_path}")
    return target


def _strip_data_url(value: str) -> tuple[str, str]:
    value = "".join(value.strip().split())
    if not value.startswith("data:"):
        return value, ""
    header, _, encoded = value.partition(",")
    if not encoded:
        raise AttachmentError("Invalid data URL attachment content")
    content_type = ""
    if header.startswith("data:"):
        metadata = header[5:]
        if metadata:
            content_type = metadata.split(";", maxsplit=1)[0]
    return encoded, content_type


async def file_to_base64(path: str) -> str:
    async with aiofiles.open(path, "rb") as f:
        content = await f.read()
    return base64.b64encode(content).decode("utf-8")


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_attachment_row(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"filename", "file_path", "url", "key"}
    return (
        required.issubset(value.keys()) and _raw_base64_content(value) is None
    )


def _field_is_multiple(field_config: dict) -> bool:
    if not isinstance(field_config, dict):
        return True
    return bool(field_config.get("multiple", False))


def _safe_path_segment(value: Any, field_name: str) -> str:
    segment = str(value or "").strip()
    if not segment or segment in {".", ".."}:
        raise AttachmentError(f"Invalid {field_name}")
    if "/" in segment or "\\" in segment:
        raise AttachmentError(f"Invalid {field_name}")
    if Path(segment).name != segment:
        raise AttachmentError(f"Invalid {field_name}")
    return segment


def _attachment_relative_path(attachment: dict) -> Path:
    file_path = str(attachment.get("file_path") or "").strip("/")
    if not file_path:
        raise AttachmentError("Missing file_path")
    parts = [
        _safe_path_segment(part, "file_path")
        for part in file_path.split("/")
        if part
    ]
    if not parts:
        raise AttachmentError("Invalid file_path")
    filename = _safe_path_segment(attachment.get("filename", ""), "filename")
    return Path(*parts, filename)


def _guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _normalize_file_fields(model: type[CoreModel] | None) -> dict[str, dict]:
    if model is None or not hasattr(model, "file_fields"):
        return {}
    file_fields = model.file_fields()
    if not isinstance(file_fields, dict):
        return {}
    return copy.deepcopy(file_fields)


class ServiceAttachment:
    def __init__(
        self, local_settings: OzonEnvCoreSettings | None = None
    ) -> None:
        self.local_settings = local_settings or OzonEnvCoreSettings.from_env()

    async def insert(
        self,
        model: type[CoreModel],
        record: dict,
        data_model: str,
        rec_name: str,
    ) -> AttachmentSavePlan:
        return await self.save_files(model, record, data_model, rec_name)

    async def update(
        self,
        model: type[CoreModel],
        record: dict,
        data_model: str,
        rec_name: str,
    ) -> AttachmentSavePlan:
        return await self.save_files(model, record, data_model, rec_name)

    async def upsert(
        self,
        model: type[CoreModel],
        record: dict,
        data_model: str,
        rec_name: str,
    ) -> AttachmentSavePlan:
        return await self.save_files(model, record, data_model, rec_name)

    async def save_files(
        self,
        model: type[CoreModel],
        record: dict,
        data_model: str,
        rec_name: str,
    ) -> AttachmentSavePlan:
        payload = copy.deepcopy(record)
        attachments_to_save: list[dict] = []
        file_fields = _normalize_file_fields(model)
        if not file_fields:
            return AttachmentSavePlan(
                record=payload,
                attachments_to_save=attachments_to_save,
            )
        safe_data_model = _safe_path_segment(data_model, "data_model")
        safe_rec_name = _safe_path_segment(rec_name, "rec_name")
        for field_key, field_config in file_fields.items():
            if field_key not in payload:
                continue
            rows, saved = await self._normalize_db_value(
                field_key=field_key,
                field_config=field_config,
                value=payload.get(field_key),
                data_model=safe_data_model,
                rec_name=safe_rec_name,
            )
            payload[field_key] = rows
            attachments_to_save.extend(saved)
        return AttachmentSavePlan(
            record=payload,
            attachments_to_save=attachments_to_save,
        )

    async def prepare_transport_files(
        self,
        model: type[CoreModel],
        record: dict,
    ) -> dict:
        payload = copy.deepcopy(record)
        file_fields = _normalize_file_fields(model)
        if not file_fields:
            return payload
        for field_key, field_config in file_fields.items():
            if field_key not in payload:
                continue
            payload[field_key] = await self._normalize_transport_value(
                field_key=field_key,
                field_config=field_config,
                value=payload.get(field_key),
            )
        return payload

    async def dump_base64_files(
        self,
        model: type[CoreModel],
        record: dict,
    ) -> dict:
        payload = copy.deepcopy(record)
        file_fields = _normalize_file_fields(model)
        if not file_fields:
            return payload
        for field_key in file_fields:
            if field_key not in payload:
                continue
            payload[field_key] = await self._normalize_dump_value(
                payload.get(field_key)
            )
        return payload

    async def _normalize_db_value(
        self,
        field_key: str,
        field_config: dict,
        value: Any,
        data_model: str,
        rec_name: str,
    ) -> tuple[list[dict], list[dict]]:
        rows: list[dict] = []
        attachments_to_save: list[dict] = []
        for item in _listify(value):
            if item in [None, ""]:
                continue
            if _is_attachment_row(item):
                rows.append(copy.deepcopy(item))
                continue
            upload = _base64_upload_from_dict(item)
            if upload is not None:
                attachment = await self._save_attachment(
                    data_model=data_model,
                    rec_name=rec_name,
                    spooled_file=upload,
                )
                rows.append(attachment)
                attachments_to_save.append(attachment)
                continue
            if isinstance(item, (str, Path)):
                attachment = await self._save_attachment_from_path(
                    data_model=data_model,
                    rec_name=rec_name,
                    source_path=Path(item).expanduser(),
                )
                rows.append(attachment)
                attachments_to_save.append(attachment)
                continue
            raise AttachmentError(
                f"Unsupported attachment value for field '{field_key}'"
            )
        self._validate_multiplicity(field_key, field_config, rows)
        return rows, attachments_to_save

    async def _normalize_transport_value(
        self,
        field_key: str,
        field_config: dict,
        value: Any,
    ) -> list[dict]:
        rows: list[dict] = []
        for item in _listify(value):
            if item in [None, ""]:
                continue
            if _is_attachment_row(item):
                rows.append(copy.deepcopy(item))
                continue
            upload = _base64_upload_from_dict(item)
            if upload is not None:
                rows.append(copy.deepcopy(item))
                continue
            if isinstance(item, (str, Path)):
                rows.append(
                    await self._path_to_base64_payload(Path(item).expanduser())
                )
                continue
            raise AttachmentError(
                f"Unsupported attachment value for field '{field_key}'"
            )
        self._validate_multiplicity(field_key, field_config, rows)
        return rows

    async def _normalize_dump_value(self, value: Any) -> list[dict]:
        rows: list[dict] = []
        for item in _listify(value):
            if not isinstance(item, dict):
                rows.append(item)
                continue
            if not _is_attachment_row(item):
                rows.append(copy.deepcopy(item))
                continue
            try:
                rows.append(await self._attachment_to_base64_payload(item))
            except AttachmentError:
                logger.warning(
                    "Cannot dump attachment as base64: %s",
                    item.get("filename", ""),
                    exc_info=True,
                )
                rows.append(copy.deepcopy(item))
        return rows

    def _validate_multiplicity(
        self,
        field_key: str,
        field_config: dict,
        items: list[dict],
    ) -> None:
        if not _field_is_multiple(field_config) and len(items) > 1:
            raise AttachmentError(
                f"Field '{field_key}' does not accept multiple files"
            )

    async def discard_attachments(
        self,
        attachments_to_save: list[dict],
    ) -> None:
        for attachment in attachments_to_save:
            path = self._tmp_upload_file(attachment)
            await asyncio.to_thread(path.unlink, missing_ok=True)

    async def download_attachment(
        self,
        data_model: str,
        uuidpath: str,
        file_name: str,
    ) -> FileResponse:
        attachment = {
            "filename": _safe_path_segment(file_name, "filename"),
            "file_path": (
                f"{_safe_path_segment(data_model, 'data_model')}/"
                f"{_safe_path_segment(uuidpath, 'uuidpath')}"
            ),
        }
        path = self._final_upload_file(attachment)
        if not path.exists() or not path.is_file():
            raise AttachmentError("Attachment not found")
        return FileResponse(
            path,
            filename=attachment["filename"],
            media_type="application/octet-stream",
        )

    async def save_attachment(
        self,
        data_model: str,
        spooled_file: Any,
        file_name_prefix: str = "",
    ) -> dict:
        rec_name = str(uuid.uuid4())
        return await self._save_attachment(
            data_model=data_model,
            rec_name=rec_name,
            spooled_file=spooled_file,
            file_name_prefix=file_name_prefix,
        )

    async def _save_attachment(
        self,
        data_model: str,
        rec_name: str,
        spooled_file: Any,
        file_name_prefix: str = "",
    ) -> dict:
        data_model = _safe_path_segment(data_model, "data_model")
        rec_name = _safe_path_segment(rec_name, "rec_name")
        file_name = _safe_path_segment(
            getattr(spooled_file, "filename", ""),
            "filename",
        )
        if file_name_prefix:
            file_name = _safe_path_segment(
                f"{file_name_prefix}_{file_name}",
                "filename",
            )
        attachment = {
            "filename": file_name,
            "content_type": (
                getattr(spooled_file, "content_type", None)
                or "application/octet-stream"
            ),
            "file_path": f"{data_model}/{rec_name}",
            "url": f"/{data_model}/{rec_name}/{file_name}",
            "key": f"{rec_name}",
        }
        tmp_file = self._tmp_upload_file(attachment)
        await asyncio.to_thread(
            tmp_file.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        await self._write_upload_file(spooled_file, tmp_file)
        return attachment

    async def _save_attachment_from_path(
        self,
        data_model: str,
        rec_name: str,
        source_path: Path,
        file_name_prefix: str = "",
    ) -> dict:
        source = source_path.expanduser().resolve(strict=False)
        if not source.exists() or not source.is_file():
            raise AttachmentError(f"Attachment path not found: {source_path}")
        file_name = _safe_path_segment(source.name, "filename")
        if file_name_prefix:
            file_name = _safe_path_segment(
                f"{file_name_prefix}_{file_name}",
                "filename",
            )
        attachment = {
            "filename": file_name,
            "content_type": _guess_content_type(source),
            "file_path": f"{_safe_path_segment(data_model, 'data_model')}/"
            f"{_safe_path_segment(rec_name, 'rec_name')}",
            "url": (
                f"/{_safe_path_segment(data_model, 'data_model')}/"
                f"{_safe_path_segment(rec_name, 'rec_name')}/{file_name}"
            ),
            "key": f"{_safe_path_segment(rec_name, 'rec_name')}",
        }
        tmp_file = self._tmp_upload_file(attachment)
        await asyncio.to_thread(
            tmp_file.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        await asyncio.to_thread(shutil.copyfile, str(source), str(tmp_file))
        return attachment

    async def _path_to_base64_payload(self, path: Path) -> dict:
        source = path.expanduser().resolve(strict=False)
        if not source.exists() or not source.is_file():
            raise AttachmentError(f"Attachment path not found: {path}")
        async with aiofiles.open(source, "rb") as handle:
            content = await handle.read()
        return {
            "filename": source.name,
            "content_type": _guess_content_type(source),
            "base64": base64.b64encode(content).decode("utf-8"),
        }

    async def _attachment_to_base64_payload(self, attachment: dict) -> dict:
        path = self._final_upload_file(attachment)
        if not path.exists() or not path.is_file():
            raise AttachmentError(f"Attachment not found: {path}")
        async with aiofiles.open(path, "rb") as handle:
            content = await handle.read()
        payload = copy.deepcopy(attachment)
        payload["base64"] = base64.b64encode(content).decode("utf-8")
        payload["content_type"] = payload.get(
            "content_type",
            _guess_content_type(path),
        )
        return payload

    async def move_attachment(self, attachment: dict) -> str:
        logger.info("save %s", attachment["filename"])
        form_upload = self._tmp_upload_file(attachment)
        to_upload_file = self._final_upload_file(attachment)
        await asyncio.to_thread(
            to_upload_file.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        if not form_upload.exists():
            raise AttachmentError(
                f"Temporary attachment not found: {form_upload}"
            )
        await asyncio.to_thread(
            shutil.move,
            str(form_upload),
            str(to_upload_file),
        )
        return str(to_upload_file)

    async def _write_upload_file(
        self, spooled_file: Any, destination: Path
    ) -> None:
        with destination.open("wb") as handle:
            while True:
                chunk = await spooled_file.read(1024 * 1024)
                if not chunk:
                    break
                await asyncio.to_thread(handle.write, chunk)

    def _tmp_upload_file(self, attachment: dict) -> Path:
        return _resolve_inside(
            Path(self.local_settings.tmp_upload_folder),
            _attachment_relative_path(attachment),
        )

    def _final_upload_file(self, attachment: dict) -> Path:
        return _resolve_inside(
            Path(self.local_settings.upload_folder),
            _attachment_relative_path(attachment),
        )


class ModelService:
    def __init__(self, model: CoreModel, orm, tz):
        self.model: CoreModel = model
        self.orm = orm
        self.tranform = (
            self.model.tranform_data_value()
            if hasattr(self.model, "tranform_data_value")
            else {}
        )
        self.dte = DateEngine(TZ=tz)
        self.data = {}
        self.data_value_service = DataValueService(
            model=self.model,
            orm=self.orm,
            dte=self.dte,
        )
        self.attachmentService = ServiceAttachment()

    def readable_float(self, val, dp=2, g=True):
        return self.data_value_service.readable_float(val, dp=dp, g=g)

    def eval_datetime(
        self,
        value: AwareDatetime,
        name: str = None,
        transform_config: dict = None,
    ):
        return self.data_value_service.eval_datetime(
            value=value,
            name=name,
            transform_config=transform_config,
        )

    def eval_float(self, value, name, transform_config: dict = None):
        return self.data_value_service.eval_float(
            value=value,
            name=name,
            transform_config=transform_config,
        )

    async def select_values(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, list]:
        return await self.data_value_service.select_values(
            select=select,
            options=options,
            value=value,
        )

    async def select_url_values(
        self, select: dict, value: Union[str, list]
    ) -> list[dict[str, Any]]:
        return await self.data_value_service.select_url_values(
            select=select, value=value
        )

    async def select_url(
        self, select: dict, options: list, value: Union[str, list]
    ) -> List[dict[str, str]]:
        return await self.data_value_service.select_url(
            select=select,
            options=options,
            value=value,
        )

    async def select_custom(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        return await self.data_value_service.select_custom(
            select=select,
            options=options,
            value=value,
        )

    async def select_resource(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        return await self.data_value_service.select_resource(
            select=select,
            options=options,
            value=value,
        )

    async def eval_select(self, name, value):
        return await self.data_value_service.eval_select(
            name=name,
            value=value,
        )

    def check_update_data_value(self, name, data_value, value):
        return self.data_value_service.check_update_data_value(
            name=name,
            data_value=data_value,
            value=value,
        )

    async def compute_data_value(self, dati: dict, pdata_value: dict = None):
        return await self.data_value_service.compute_data_value(
            dati=dati,
            pdata_value=pdata_value,
        )
