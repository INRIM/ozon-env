# Copyright INRIM (https://www.inrim.eu)
# See LICENSE file for full licensing details.
from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import operator
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, time
from functools import reduce
from pathlib import Path
from typing import Any, Optional, get_origin, get_args
from typing import TypeVar, Generic, List, Dict
from zoneinfo import ZoneInfo

from bson import Decimal128, Int64
from dateutil.parser import parse
from ozonenv.core.db.BsonTypes import PyObjectId, bson, BsonEncoder
from ozonenv.core.utils import unwrap_optional
from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    PrivateAttr,
    field_serializer,
)

defaultdt = '1970-01-01T00:00:00+00:00'

logger = logging.getLogger("asyncio")

T = TypeVar("T", bound=BaseModel)
D = TypeVar("D")
ModelType = TypeVar("ModelType", bound=BaseModel)

default_fields = [
    "owner_uid",
    "owner_name",
    "owner_function",
    "owner_sector",
    "create_datetime",
    "update_uid",
    "update_datetime",
    "owner_personal_type",
    "owner_job_title",
    "owner_function_type",
    "owner_mail",
]

list_default_fields_update = [
    "create_datetime",
    "update_uid",
    "update_datetime",
]

data_fields = ["data", "data_value"]

default_data_fields = default_fields + data_fields

default_data_fields_update = list_default_fields_update + data_fields

default_list_metadata = [
    "id",
    "rec_name",
    "owner_uid",
    "owner_name",
    "owner_sector",
    "owner_sector_id",
    "owner_function",
    "update_datetime",
    "create_datetime",
    "owner_mail",
    "owner_function_type",
    "childs",
    "update_uid",
    "app_code",
    "parent",
    "process_id",
    "data_value",
    "sys",
    "demo",
    "deleted",
    "list_order",
    "owner_personal_type",
    "owner_job_title",
]
default_list_metadata_clean = [
    "id",
    "rec_name",
    "owner_uid",
    "owner_name",
    "owner_sector",
    "owner_sector_id",
    "owner_function",
    "update_datetime",
    "create_datetime",
    "owner_mail",
    "owner_function_type",
    "childs",
    "update_uid",
    "app_code",
    "parent",
    "process_id",
    "sys",
    "demo",
    "deleted",
    "list_order",
    "owner_personal_type",
    "owner_job_title",
]

default_list_metadata_fields = [
    "id",
    "owner_uid",
    "owner_name",
    "owner_sector",
    "owner_sector_id",
    "owner_function",
    "update_datetime",
    "create_datetime",
    "owner_mail",
    "update_uid",
    "owner_function_type",
    "sys",
    "demo",
    "deleted",
    "list_order",
    "owner_personal_type",
    "owner_job_title",
]

default_list_metadata_fields_update = [
    "id",
    "rec_name",
    "owner_uid",
    "owner_name",
    "owner_sector",
    "owner_sector_id",
    "owner_function",
    "create_datetime",
    "owner_mail",
    "owner_personal_type",
    "owner_job_title",
]

export_list_metadata = [
    "owner_uid",
    "owner_name",
    "owner_function",
    "owner_sector",
    "owner_sector_id",
    "owner_personal_type",
    "owner_job_title",
    "owner_function_type",
    "create_datetime",
    "update_uid",
    "update_datetime",
    "list_order",
    "owner_mail",
    "sys",
]


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DbViewModel(BaseModel):
    name: str
    model: str
    force_recreate: bool = False
    pipeline: list


class MainModel(BaseModel):
    _file_dump_mode: str = PrivateAttr(default="attachment")
    _file_dump_upload_folder: str = PrivateAttr(default="")

    @classmethod
    def str_name(cls, *args, **kwargs):
        return cls.model_json_schema(*args, **kwargs).get("title", "").lower()

    def get_dict(self, exclude=None, compute_datetime: bool = True):
        if exclude is None:
            exclude = []
        basic = ["status", "message", "res_data", "session_diff", "tz"]
        d = self.model_copy(deep=True).model_dump(
            exclude=set().union(basic, exclude)
        )
        if self._file_dump_mode == "base64":
            d = self._dump_file_fields_as_base64(d)
        return d

    def enable_base64_file_dump(self, upload_folder: str) -> None:
        self._file_dump_mode = "base64"
        self._file_dump_upload_folder = str(upload_folder or "").strip()

    def disable_base64_file_dump(self) -> None:
        self._file_dump_mode = "attachment"
        self._file_dump_upload_folder = ""

    def _dump_file_fields_as_base64(self, data: dict) -> dict:
        if not self._file_dump_upload_folder:
            return data
        file_fields = {}
        if hasattr(self.__class__, "file_fields"):
            file_fields = self.__class__.file_fields() or {}
        if not isinstance(file_fields, dict) or not file_fields:
            return data
        upload_root = Path(self._file_dump_upload_folder).expanduser()
        for field_key in file_fields.keys():
            if field_key not in data:
                continue
            data[field_key] = self._dump_attachment_value_as_base64(
                data.get(field_key),
                upload_root,
            )
        return data

    def _dump_attachment_value_as_base64(
        self,
        value: Any,
        upload_root: Path,
    ) -> list[Any]:
        items = value if isinstance(value, list) else [value]
        dumped: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                dumped.append(item)
                continue
            required = {"filename", "file_path", "url", "key"}
            if not required.issubset(item.keys()):
                dumped.append(copy.deepcopy(item))
                continue
            relative_dir = str(item.get("file_path") or "").strip("/")
            file_name = str(item.get("filename") or "").strip()
            if not relative_dir or not file_name:
                dumped.append(copy.deepcopy(item))
                continue
            file_path = upload_root / relative_dir / file_name
            if not file_path.exists() or not file_path.is_file():
                dumped.append(copy.deepcopy(item))
                continue
            payload = copy.deepcopy(item)
            payload["content_type"] = payload.get("content_type") or (
                mimetypes.guess_type(file_path.name)[0]
                or "application/octet-stream"
            )
            payload["base64"] = base64.b64encode(
                file_path.read_bytes()
            ).decode("utf-8")
            dumped.append(payload)
        return dumped

    def get_dict_json(self, exclude=[]):
        return json.loads(
            json.dumps(
                self.get_dict(exclude=exclude),
                cls=BsonEncoder,
                ensure_ascii=False,
            )
        )

    def get_dict_copy(self):
        return copy.deepcopy(self.get_dict())

    async def dump_model_dict_async(self) -> dict:
        return await asyncio.to_thread(self.get_dict)

    def get_dict_diff(
        self, to_compare_dict, ignore_fields=[], remove_ignore_fileds=True
    ):
        """
        deprecated but works use model.get_dict_diff() to get dict diff
        :param to_compare_dict:
        :param ignore_fields:
        :param remove_ignore_fileds:
        :return:
        """
        if ignore_fields and remove_ignore_fileds:
            original_dict = self.get_dict(exclude=ignore_fields)
        else:
            original_dict = self.get_dict()
        diff = {
            k: v
            for k, v in to_compare_dict.items()
            if k in original_dict and not original_dict[k] == v
        }
        return diff.copy()

    def scan_data(self, key, default=None):
        data = self.get_dict(exclude=["_id", "id"])
        try:
            _keys = key.split(".")
            keys = []
            for v in _keys:
                if str(v).isdigit():
                    keys.append(int(v))
                else:
                    keys.append(v)
            lastplace = reduce(operator.getitem, keys[:-1], dict(data))
            return lastplace.get(keys[-1], default)
        except Exception as e:
            print(f" error scan_data {e} field not found")
            return default

    def get(self, val, default: Optional = None):
        try:
            if "." in val:
                return self.scan_data(val, default)
            elif default:
                return getattr(self, val, default)
            else:
                return getattr(self, val)
        except Exception as e:
            logger.error(
                f" error  {e} field {val} not found return default",
                exc_info=True,
            )
            return default

    def set_from_child(self, key, nodes: str, default):
        # old = getattr(self, key)
        new = self.get(nodes, default)
        setattr(self, key, new)
        # self.on_field_change(key, old, new)

    def set(self, key, value):
        # old = getattr(self, key)
        setattr(self, key, value)
        # self.on_field_change(key, old, value)

    def addfile(self, field_key: str, value: Any) -> None:
        file_fields = {}
        if hasattr(self.__class__, "file_fields"):
            file_fields = self.__class__.file_fields() or {}
        field_config = file_fields.get(field_key)
        if not isinstance(field_config, dict):
            raise ValueError(f"Field '{field_key}' is not a file field")
        items = value if isinstance(value, list) else [value]
        items = [item for item in items if item not in [None, ""]]
        is_multiple = bool(field_config.get("multiple", False))
        if not is_multiple and len(items) > 1:
            raise ValueError(
                f"Field '{field_key}' does not accept multiple files"
            )
        if is_multiple:
            current = getattr(self, field_key, None)
            current_items = (
                current
                if isinstance(current, list)
                else [current] if current not in [None, ""] else []
            )
            current_items.extend(items)
            setattr(self, field_key, current_items)
            return
        setattr(self, field_key, items[:1] if items else [])

    def add_text(self, key, value: str, prefix: str = ""):
        val = getattr(self, key)
        # old = val
        if val and not val == "":
            val = f"{val}, {value}"
            if prefix:
                val = f"{prefix} {val}"
            setattr(self, key, val)
            # self.on_field_change(key, old, val)
        else:
            val = value
            if prefix:
                val = f"{prefix} {val}"
            setattr(self, key, val)
            # self.on_field_change(key, old, val)

    def set_many(self, data_dict):
        for k, v in data_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
                # self.on_field_change(k, old, v)

    def selection_value(self, key, value, read_value):
        # old = getattr(self, key)
        setattr(self, key, value)
        # self.on_field_change(key, old, value)
        self.data_value[key] = read_value

    def selection_value_from_record(self, key, src, src_key=""):
        if not src_key:
            src_key = key
        # old = getattr(self, key)
        val = getattr(src, src_key)
        setattr(self, key, val)
        self.data_value[key] = src.data_value[src_key]

    @classmethod
    def iso_to_utc(cls, date_str) -> datetime:
        # date_str es: "2025-04-09T00:00:00+02:00" oppure "2025-04-09T00:00:00Z" o con offset
        # 1. parse ISO
        dt = datetime.fromisoformat(date_str)
        # dt è aware se date_str contiene offset, altrimenti naive (attenzione)
        if dt.tzinfo is None:
            # interpreta naive come locale dell’app (opzionale)
            # oppure rigetta l’input
            dt = datetime.fromisoformat(f"{date_str}+00:00")
        # 2. converti a UTC
        dt_utc = dt.astimezone(ZoneInfo("UTC"))
        return dt_utc

    @classmethod
    def iso_to_utc_str(cls, date_str: str) -> str:
        return cls.iso_to_utc(date_str).isoformat()

    @classmethod
    def utc_now(cls) -> AwareDatetime:
        return datetime.now(ZoneInfo("UTC"))

    @classmethod
    def default_datetime(cls) -> datetime:
        return cls.iso_to_utc(defaultdt)

    @classmethod
    def datetime_fields(cls):
        return {}

    @classmethod
    def nested_datetime_fields(cls):
        return {}

    @classmethod
    def nested_transform_data_value(cls):
        return {}

    @classmethod
    def normalize_datetime_fields(cls, tz: str, dati: dict) -> dict:
        """
        Controlla tutti i campi datetime del model (inclusi nested model):
          - se il valore è naive, assume che sia in self.tz
          - lo converte in UTC e aggiorna il dizionario
        Ritorna il dizionario modificato
        """
        tz_base = ZoneInfo(tz)

        def _normalize_model_fields(
            model: type[MainModel], mdata: dict, nested_field: str = None
        ) -> dict:
            for name, field in model.model_fields.items():

                if name not in mdata:
                    continue

                raw_value = mdata[name]
                actual_type = unwrap_optional(field.annotation)

                # --- Single nested Pydantic model ---
                if isinstance(raw_value, dict) and hasattr(
                    actual_type, "model_fields"
                ):
                    nested_result = _normalize_model_fields(
                        actual_type, raw_value, name
                    )
                    mdata[name] = nested_result
                    continue

                # --- List/Tuple of nested Pydantic models ---
                origin = get_origin(actual_type)
                args = get_args(actual_type)
                if origin in (list, tuple) and args:
                    elem_type = unwrap_optional(args[0])
                    if isinstance(raw_value, list) and hasattr(
                        elem_type, "model_fields"
                    ):
                        # datagrid
                        for idx, item in enumerate(raw_value):
                            if isinstance(item, dict):
                                el_data = _normalize_model_fields(
                                    elem_type, item, name
                                )
                                mdata[name][idx] = el_data
                        continue

                if field.annotation in (
                    datetime,
                    AwareDatetime,
                    Optional[AwareDatetime],
                ):
                    if nested_field:
                        dttype = (
                            cls.nested_datetime_fields()
                            .get(nested_field, {})
                            .get(name, {})
                            .get("transform", {})
                            .get("type", "datetime")
                        )
                    else:
                        dttype = (
                            model.datetime_fields()
                            .get(name, {})
                            .get("transform", {})
                            .get("type", "datetime")
                        )

                    if raw_value is None:
                        continue

                    # parsing
                    if isinstance(raw_value, str):
                        try:
                            value = datetime.fromisoformat(raw_value)
                        except ValueError:
                            value = BasicModel.default_datetime()
                    elif isinstance(raw_value, datetime):
                        value = raw_value
                    elif isinstance(raw_value, date):
                        value = datetime.combine(raw_value, time.min)
                    else:
                        continue

                    # --- CASO DATE ---
                    if dttype == "date":
                        value = datetime(
                            value.year,
                            value.month,
                            value.day,
                            tzinfo=ZoneInfo("UTC"),
                        )
                        mdata[name] = value
                        continue

                    # --- CASO DATETIME ---
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=tz_base)

                    utc_value = value.astimezone(ZoneInfo("UTC"))
                    mdata[name] = utc_value
                elif field.annotation in [int, Optional[int]]:
                    if type(raw_value) is str:
                        try:
                            mdata[name] = int(raw_value)
                        except ValueError:
                            mdata[name] = 0
                elif field.annotation in [float, Optional[float]]:
                    if type(raw_value) in [Decimal128, str]:
                        try:
                            mdata[name] = float(str(raw_value))
                        except ValueError:
                            mdata[name] = 0.0
                elif field.annotation in [int, Optional[int]]:
                    if type(raw_value) in [Int64, str]:
                        try:
                            mdata[name] = int(str(raw_value))
                        except ValueError:
                            mdata[name] = 0
            return mdata

        return _normalize_model_fields(cls, dati)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "alias_generator": lambda f_name: f_name.replace(".", "_"),
        "tz_aware": True,
        "ignored_types": (type(BaseModel),),
    }


class CoreNestedModel(MainModel):
    data_value: dict = Field(default_factory=dict)


class CoreModel(MainModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    data_model: str = ""
    rec_name: str = ""
    app_code: str | None = None
    parent: str = ""
    process_id: str = ""
    process_task_id: str = ""
    data_value: dict = Field(default_factory=dict)
    owner_name: str = ""
    deleted: float = 0
    list_order: int = 0
    owner_uid: str = ""
    owner_mail: str = ""
    owner_function: str = ""
    owner_function_type: str = ""
    owner_sector: str = ""
    owner_sector_id: int = 0
    owner_personal_type: str = ""
    owner_job_title: str = ""
    update_uid: str = ""
    sys: bool = False
    default: bool = False
    active: bool = True
    demo: bool = False
    childs: List[Dict] = Field(default=[])
    create_datetime: AwareDatetime = Field(default=MainModel.utc_now())
    update_datetime: AwareDatetime = Field(
        default=MainModel.iso_to_utc(defaultdt)
    )
    status: str = "ok"
    message: str = ""
    res_data: dict = Field(default={})
    session_diff: dict = Field(default={})
    tz: str = "Europe/Rome"

    @field_serializer('id')
    def serialize_dt(self, id: PyObjectId, _info):
        return str(id)

    @classmethod
    def str_name(cls, *args, **kwargs):
        return cls.model_json_schema(*args, **kwargs).get("title", "").lower()

    def renew_id(self):
        self.id = PyObjectId()

    def get_dict_copy(self, exclude=[], compute_datetime: bool = True):
        return self.get_dict(
            exclude=exclude, compute_datetime=compute_datetime
        )

    def rec_name_domain(self):
        return {"rec_name": self.rec_name}.copy()

    def id_domain(self):
        return {"_id": bson.ObjectId(self.id)}.copy()

    def reset_diff(self):
        self.session_diff = {}

    def is_error(self):
        return self.status == "error"

    def is_to_delete(self):
        return self.deleted > 0

    def set_active(self):
        self.deleted = 0
        self.active = True

    def set_archive(self):
        self.deleted = 0
        self.active = False

    def set_to_delete(self, timestamp):
        self.deleted = timestamp
        self.active = False

    def set_list_order(self, val):
        self.list_order = val

    @classmethod
    def get_value_for_select_list(cls, list_src, key, label_key="label"):
        for item in list_src:
            if item.get("value") == key:
                return item.get(label_key)
        return ""

    def selection_value_resources(
        self, key: str, value: str, resources: list, label_key: str = "label"
    ):
        '''

        :param key:
        :param value:
        :param resources:
        :param label_key:
        :return:
        '''
        value_label = self.get_value_for_select_list(
            resources, value, label_key=label_key
        )
        self.selection_value(key, value, value_label)

    @classmethod
    def no_clone_field_keys(cls):
        return ["list_order"]

    def clone_data(self):
        dat = self.get_dict_copy(exclude=self.no_clone_field_keys())
        return dat.copy()

    def to_datetime(self, key):
        """
        DEPRECATED
        :param key:
        :return:
        """
        v = self.get(key)
        try:
            return parse(v)
        except Exception:
            return v

    @classmethod
    def schema(cls):
        return {}

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def tranform_data_value(cls):
        return {}

    @classmethod
    def fields_limit_value(cls):
        return {}

    @classmethod
    def create_task_action(cls):
        return []

    @classmethod
    def fields_properties(cls):
        return {}

    @classmethod
    def default_hidden_fields(cls):
        return []

    @classmethod
    def default_readonly_fields(cls):
        return []

    @classmethod
    def default_disabled_fields(cls):
        return []

    @classmethod
    def default_required_fields(cls):
        return []

    @classmethod
    def realted_fields_logic(cls):
        return {}

    @classmethod
    def fields_logic(cls):
        return {}

    @classmethod
    def fields_conditional(cls):
        return {}

    @classmethod
    def filter_keys(cls):
        return []

    @classmethod
    def select_fields(cls):
        return {}

    @classmethod
    def select_options(cls, key: str = None, update_options: dict = None):
        options = {}
        if key and update_options and key in options:
            options[key] = update_options.copy()
        return options.copy()

    @classmethod
    def get_data_model(cls):
        return ""

    @classmethod
    def get_version(cls):
        return ""

    @classmethod
    def model_depends(cls):
        return []

    async def dump_model_async(self) -> str:
        """Esegue model_dump_json() in modo async-safe."""
        # model_dump_json è CPU-bound, quindi usiamo to_thread per non bloccare l'event loop
        return await asyncio.to_thread(self.model_dump_json)


class BasicModel(CoreModel):
    @classmethod
    def get_unique_fields(cls) -> []:
        return ["rec_name"]

    @classmethod
    def computed_fields(cls) -> {}:
        return {}

    @classmethod
    def no_clone_field_keys(cls) -> []:
        return ["rec_name", "list_order"]

    @classmethod
    def config_fields(cls) -> {}:
        return {}

    @classmethod
    def table_columns(cls) -> dict:
        return {}

    @classmethod
    def file_fields(cls) -> dict:
        return {}


class User(BasicModel):
    uid: str
    password: str = Field(default="", exclude=True)
    token: dict[str, Any] | str = Field(default_factory=dict)
    req_id: str = ""
    parent: str = ""
    childs: list[Any] = Field(default_factory=list)
    last_update: int | str | Decimal128 = 0
    is_admin: bool = False
    is_bot: bool = False
    is_api: bool = False
    use_auth: bool = True
    rec_name: str = ""
    nome: str = ""
    cognome: str = ""
    mail: str = ""
    matricola: str = ""
    codicefiscale: str = ""
    data_value: dict[str, Any] = Field(default_factory=dict)
    allowed_users: list[str] = Field(default_factory=list)
    user_data: dict[str, Any] = Field(default_factory=dict)
    list_order: int = 1
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    user_function: str = ""
    function: str = ""
    owner_function: str = ""
    owner_sector: Optional[str] = ""
    owner_mail: Optional[str] = ""
    owner_sector_id: Optional[int] = 0
    owner_personal_type: Optional[str] = ""
    owner_job_title: Optional[str] = ""
    create_datetime: Optional[datetime] = None
    update_datetime: Optional[datetime] = None
    sector: Optional[str] = ""
    sector_id: Optional[int] = 0
    sector_code: Optional[str] = ""
    last_login: Optional[datetime] = None
    sys: bool = False
    active: bool = True
    default: bool = True
    demo: bool = False
    tz: str = "Europe/Rome"
    user_role: list[str] = Field(default_factory=lambda: ["base"])
    tech_admin: bool = False
    groups: list[str] = Field(default_factory=list)
    full_name: str = ""
    is_public: bool = False
    claims: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    client_id: str = ""

    @classmethod
    def get_unique_fields(cls):
        return ["rec_name", "uid"]


class JobContext(BasicModel):
    job_token: str
    client_id: str
    job_key: str
    process_instance_key: str
    resolved_user_id: str
    issued_at: datetime
    expires_at: datetime

    @classmethod
    def get_unique_fields(cls):
        return ["job_token"]

    @classmethod
    def datetime_fields(cls):
        return {
            "issued_at": {
                "transform": {
                    "type": "datetime",
                }
            },
            "expires_at": {
                "transform": {
                    "type": "datetime",
                }
            },
            "create_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
            "update_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
        }


class AttachmentTrash(BasicModel):
    parent: str = ""
    model: str = ""
    # modell_ because model_rec_name has
    # conflict with protected namespace "model_".
    modell_rec_name: str = ""
    attachments: List[Dict] = []

    @classmethod
    def datetime_fields(self):
        return {
            "create_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
            "update_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
        }


class Component(BasicModel):
    title: str = ""
    path: str = ""
    parent: str = ""
    parent_name: str = ""
    components: List[dict] = []
    links: Dict = {}
    type: str = "form"
    no_cancel: int = 0
    display: str = ""
    action: str = ""
    tags: Optional[List[str]] = []
    settings: Dict = {}
    properties: Dict = {}
    handle_global_change: int = 1
    process_tenant: str = ""
    make_virtual_model: bool = False
    authenticate: bool = True
    projectId: str = ""  # needed for compatibility with fomriojs

    @classmethod
    def get_unique_fields(cls):
        return ["rec_name", "title"]

    @classmethod
    def datetime_fields(self):
        return {
            "create_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
            "update_datetime": {
                "transform": {
                    "type": "datetime",
                }
            },
        }


class DictRecord(BaseModel):
    model: str
    rec_name: str = ""
    data: dict = {}

    def __init__(self, **data):
        super().__init__(**data)
        if not self.data.get("data_value"):
            self.data["data_value"] = {}
        else:
            self.data["rec_name"] = self.rec_name

    @property
    def data_value(self):
        return self.data.get("data_value", {})

    def parse_value(self, v):
        type_def = {
            "int": int,
            "string": str,
            "float": float,
            "dict": dict,
            "list": list,
            "datetime": datetime,
        }
        s = v
        if not isinstance(v, str):
            s = str(v)
        regex = re.compile(
            r"(?P<dict>\{[^{}]+\})|(?P<list>\[[^]]+\])|(?P<float>\d*\.\d+)"
            r"|(?P<int>\d+)|(?P<string>[a-zA-Z]+)"
        )
        regex_dt = re.compile(r"(\d{4}-\d{2}-\d{2})[A-Z]+(\d{2}:\d{2}:\d{2})")
        dtr = regex_dt.search(s)
        if dtr:
            return parse(dtr.group(0))
        else:
            rgx = regex.search(s)
            if not rgx:
                return s
            if s in ["false", "true"]:
                return bool("true" == s)
            if rgx.lastgroup not in ["list", "dict"]:
                types_d = []
                for match in regex.finditer(s):
                    types_d.append(match.lastgroup)
                if len(types_d) > 1:
                    return s
                else:
                    return type_def.get(rgx.lastgroup)(s)
            else:
                return json.load(s)

    def value_type(self, v):
        type_def = {
            "int": int,
            "string": str,
            "float": float,
            "dict": dict,
            "list": list,
            "date": datetime,
        }
        s = v
        if not isinstance(v, str):
            s = str(v)
        regex = re.compile(
            r"(?P<dict>\{[^{}]+\})|(?P<list>\[[^]]+\])|(?P<float>\d*\.\d+)"
            r"|(?P<int>\d+)|(?P<string>[a-zA-Z]+)"
        )
        regex_dt = re.compile(r"(\d{4}-\d{2}-\d{2})[A-Z]+(\d{2}:\d{2}:\d{2})")
        dtr = regex_dt.search(s)
        if dtr:
            return datetime
        else:
            rgx = regex.search(s)
            if not rgx:
                return str
            if s in ["false", "true"]:
                return bool
            types_d = []
            for match in regex.finditer(s):
                types_d.append(match.lastgroup)
            if len(types_d) > 1:
                return str
            else:
                return type_def.get(rgx.lastgroup)

    def selection_value(self, key, value, read_value):
        self.data[key] = value
        self.data["data_value"][key] = read_value

    def selection_value_from_record(self, key, src, src_key=""):
        if not src_key:
            src_key = key
        self.data[key] = src.data[src_key]
        self.data["data_value"][key] = src.data["data_value"][src_key]

    def get_dict(self):
        return json.loads(self.model_dump_json())

    def rec_name_domain(self):
        return {"rec_name": self.rec_name}.copy()

    def set_active(self, user_name="admin"):
        self.data["deleted"] = 0
        self.data["active"] = True
        self.data["owner_uid"] = user_name
        self.data["list_order"] = 0
        if "data_value" not in self.data:
            self.data["data_value"] = {}

    def set_list_order(self, val):
        self.data["list_order"] = val

    def scan_data(self, key, default=None):
        try:
            _keys = key.split(".")
            keys = []
            for v in _keys:
                if str(v).isdigit():
                    keys.append(int(v))
                else:
                    keys.append(v)
            lastplace = reduce(operator.getitem, keys[:-1], self.data)
            return lastplace.get(keys[-1], default)
        except Exception:
            return default

    def get(self, val, default: Optional = None):
        if "." in val:
            return self.scan_data(val, default)
        if default:
            return self.data.get(val, default)
        else:
            return self.data.get(val)

    def set(self, key, val, pase_data=True):
        if pase_data:
            self.data[key] = self.parse_value(val)
        else:
            self.data[key] = val

    def set_from_child(self, key, nodes: str, default):
        self.data[key] = self.get(nodes, default)

    def update_field_type_value(self, key):
        val = self.data.get(key, "")
        self.data[key] = self.parse_value(val)

    def set_many(self, data_dict):
        self.data.update(data_dict)

    def get_value_for_select_list(self, list_src, key, label_key="label"):
        for item in list_src:
            if item.get("value") == key:
                return item.get(label_key)
        return ""

    def selection_value_resources(
        self, key, value, list_src, label_key="label"
    ):
        value_label = self.get_value_for_select_list(
            list_src, value, label_key=label_key
        )
        self.selection_value(key, value, value_label)

    def to_date(self, key):
        v = self.get(key)
        if self.value_type(v) is datetime:
            return parse(v)
        return v

    def clone_data(self):
        dat = copy.deepcopy(self.data)
        dat.pop("rec_name")
        dat.pop("list_order")
        return dat.copy()


class BasicReturn(BaseModel):
    fail: bool = False
    msg: str = ""
    data: dict = {}


class Settings(BasicModel):
    list_order: Optional[int] = Field(0, title='List Order')
    rec_name: Optional[str] = Field('', title='Rec Name')
    internal_port: Optional[int] = Field(0, title='Internal Port')
    app_origin_type: Optional[str] = Field('', title='App Origin Type')
    module_label: Optional[str] = Field('', title='Module Label')
    description: Optional[str] = Field('', title='Description')
    admins: Optional[List[str]] = Field([], title='Admins')
    module_type: Optional[str] = Field('app', title='Module Type')
    module_group: Optional[str] = Field('', title='Module Group')
    version: Optional[str] = Field('1.0.0', title='Version')
    port: Optional[int] = Field(0, title='Port')
    stato: Optional[str] = Field('', title='Stato')
    upload_folder: Optional[str] = Field('/uploads', title='Upload Folder')
    web_concurrency: Optional[int] = Field(1, title='Web Concurrency')
    delete_record_after_days: Optional[int] = Field(
        1, title='Delete Record After Days'
    )
    token_expire_hours: Optional[int] = Field(12, title='Token Expire Hours')
    theme: Optional[str] = Field('italia', title='Theme')
    logo_img_url: Optional[str] = Field('', title='Logo Img Url')
    server_datetime_mask: Optional[str] = Field(
        '%Y-%m-%dT%H:%M:%S', title='Server Datetime Mask'
    )
    server_date_mask: Optional[str] = Field(
        '%Y-%m-%dT%H:%M:%S', title='Server Date Mask'
    )
    ui_datetime_mask: Optional[str] = Field(
        '%d/%m/%Y %H:%M:%S', title='Ui Datetime Mask'
    )
    ui_date_mask: Optional[str] = Field('%d/%m/%Y', title='Ui Date Mask')
    tz: Optional[str] = Field('Europe/Rome', title='Tz')
    report_orientation: Optional[str] = Field(
        'Portrait', title='Report Orientation'
    )
    report_page_size: Optional[str] = Field('A4', title='Report Page Size')
    report_footer_company: Optional[str] = Field(
        '', title='Report Footer Company'
    )
    report_footer_title1: Optional[str] = Field(
        '', title='Report Footer Title1'
    )
    report_footer_sub_title: Optional[str] = Field(
        '', title='Report Footer Sub Title'
    )
    report_footer_pagination: Optional[bool] = Field(
        True, title='Report Footer Pagination'
    )
    report_header_space: Optional[str] = Field(
        '30mm', title='Report Header Space'
    )
    report_footer_space: Optional[str] = Field(
        '8mm', title='Report Footer Space'
    )
    report_margin_left: Optional[str] = Field(
        '10mm', title='Report Margin Left'
    )
    report_margin_right: Optional[str] = Field(
        '10mm', title='Report Margin Right'
    )

    @classmethod
    def get_version(cls):
        return '2022-08-01T10:11:04.635610'

    @classmethod
    def get_unique_fields(cls):
        return ['rec_name']

    @classmethod
    def computed_fields(cls):
        return {}

    @classmethod
    def no_clone_field_keys(cls):
        return ['rec_name']

    @classmethod
    def tranform_data_value(cls):
        return {}

    @classmethod
    def fields_limit_value(cls):
        return {}

    @classmethod
    def create_task_action(cls):
        return []

    @classmethod
    def fields_properties(cls):
        return {'admins': {'label': 'full_name', 'id': 'uid'}}

    @classmethod
    def default_hidden_fields(cls):
        return []

    @classmethod
    def default_readonly_fields(cls):
        return []

    @classmethod
    def default_required_fields(cls):
        return [
            'rec_name',
            'internal_port',
            'app_origin_type',
            'module_label',
            'description',
            'module_type',
            'module_group',
            'version',
            'port',
            'theme',
            'server_datetime_mask',
            'server_date_mask',
            'ui_datetime_mask',
            'tz',
            'report_orientation',
            'report_page_size',
            'report_header_space',
            'report_footer_space',
            'report_margin_left',
            'report_margin_right',
        ]

    @classmethod
    def filter_keys(cls):
        return [
            'list_order',
            'rec_name',
            'internal_port',
            'app_origin_type',
            'module_label',
            'description',
            'admins',
            'module_type',
            'module_group',
            'version',
            'port',
            'stato',
            'upload_folder',
            'web_concurrency',
            'delete_record_after_days',
            'token_expire_hours',
            'theme',
            'logo_img_url',
            'server_datetime_mask',
            'server_date_mask',
            'ui_datetime_mask',
            'ui_date_mask',
            'tz',
            'report_orientation',
            'report_page_size',
            'report_footer_company',
            'report_footer_title1',
            'report_footer_sub_title',
            'report_footer_pagination',
            'report_header_space',
            'report_footer_space',
            'report_margin_left',
            'report_margin_right',
            'domain',
            'external_proxy_uri_configs',
        ]

    @classmethod
    def config_fields(cls):
        return {
            'list_order': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'rec_name': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'internal_port': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'app_origin_type': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': '',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [
                    {'label': 'System', 'value': 'system'},
                    {'label': 'Virtual', 'value': 'virtual'},
                ],
                'template_label_keys': [],
            },
            'module_label': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'description': {
                'ctype': 'textarea',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'admins': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': '',
                'multiple': True,
                'dataSrc': 'url',
                'idPath': '',
                'resource_id': '',
                'values': [],
                'url': 'https://people.ininrim.it/api'
                '/get_addressbook_service_user/0',
                'template_label_keys': [],
            },
            'module_type': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': 'app',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [
                    {'label': 'App', 'value': 'app'},
                    {'label': 'Backend', 'value': 'server'},
                ],
                'template_label_keys': [],
            },
            'module_group': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'version': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'port': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'stato': {
                'ctype': 'select',
                'disabled': True,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': '',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [
                    {'label': 'Attivo', 'value': 'live'},
                    {'label': 'Spento', 'value': 'spento'},
                ],
                'template_label_keys': [],
            },
            'upload_folder': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'web_concurrency': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'delete_record_after_days': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'token_expire_hours': {
                'ctype': 'number',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'theme': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': 'italia',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [{'label': 'Italia', 'value': 'italia'}],
                'template_label_keys': [],
            },
            'logo_img_url': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'server_datetime_mask': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'server_date_mask': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'ui_datetime_mask': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'ui_date_mask': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'tz': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_orientation': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': 'Portrait',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [
                    {'label': 'Portrait', 'value': 'Portrait'},
                    {'label': 'Landscape', 'value': 'Landscape'},
                ],
                'template_label_keys': [],
            },
            'report_page_size': {
                'ctype': 'select',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'selectComponent',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
                'valueProperty': None,
                'selectValues': None,
                'defaultValue': 'A4',
                'multiple': False,
                'dataSrc': 'values',
                'idPath': '',
                'resource_id': '',
                'values': [
                    {'label': 'Legal', 'value': 'Legal'},
                    {'label': 'Letter', 'value': 'Letter'},
                    {'label': 'A4', 'value': 'A4'},
                    {'label': 'A3', 'value': 'A3'},
                ],
                'template_label_keys': [],
            },
            'report_footer_company': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_footer_title1': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_footer_sub_title': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_footer_pagination': {
                'ctype': 'checkbox',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_header_space': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_footer_space': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_margin_left': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'report_margin_right': {
                'ctype': 'textfield',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': True,
                'unique': False,
                'component': 'Component',
                'calculateServer': None,
                'action_type': False,
                'no_clone': False,
                'transform': {},
                'datetime': False,
                'min': False,
                'max': False,
            },
            'external_proxy_uri_configs': {},
        }

    @classmethod
    def components_ext_data_src(cls):
        return ['admins']

    @classmethod
    def get_data_model(cls):
        return ""


@dataclass
class DataReturn(Generic[D]):
    data: D | None = None
    fail: bool = False
    msg: str = ""


@dataclass(frozen=True)
class OzonEnvCoreSettings:
    app_code: str | None = None
    mongo_user: str | None = None
    mongo_pass: str | None = None
    mongo_url: str | None = None
    mongo_db: str | None = None
    mongo_replica: str | None = None
    models_folder: str | None = None
    api_prefix: str = "/v2"
    require_auth: bool = True
    auth_mode: str = "session"
    oauth_url: str | None = None
    oauth_client_id: str | None = None
    token_audience: str | None = None
    keycloak_jwks_url_value: str | None = None
    keycloak_issuer_value: str | None = None
    keycloak_algorithms: str = "RS256"
    bootstrap_user_file: str = "base_data/user.json"
    upload_folder: str = "/tmp/ozon-env-api/uploads"
    tmp_upload_folder: str = "/tmp"
    backend_interface: str = "db"

    @classmethod
    def from_env(cls) -> "OzonEnvCoreSettings":
        return cls(
            app_code=os.getenv("APP_CODE"),
            mongo_user=os.getenv("MONGO_USER"),
            mongo_pass=os.getenv("MONGO_PASS"),
            mongo_url=os.getenv("MONGO_URL"),
            mongo_db=os.getenv("MONGO_DB"),
            mongo_replica=os.getenv("MONGO_REPLICA"),
            models_folder=os.getenv("MODELS_FOLDER"),
            api_prefix=os.getenv("OZON_API_PREFIX", "/v2"),
            require_auth=_read_bool("OZON_API_REQUIRE_AUTH", True),
            auth_mode=os.getenv("OZON_AUTH_MODE", "session"),
            oauth_url=os.getenv("OZON_OAUTH_URL"),
            oauth_client_id=os.getenv("OZON_CLIENT_ID"),
            token_audience=os.getenv("OZON_TOKEN_AUDIENCE") or None,
            keycloak_jwks_url_value=os.getenv("OZON_KEYCLOAK_JWKS_URL"),
            keycloak_issuer_value=os.getenv("OZON_KEYCLOAK_ISSUER"),
            keycloak_algorithms=os.getenv("OZON_KEYCLOAK_ALGORITHMS", "RS256"),
            bootstrap_user_file=os.getenv(
                "OZON_BOOTSTRAP_USER_FILE",
                "base_data/user.json",
            ),
            upload_folder=(
                os.getenv("OZON_UPOLOAD_FOLDER")
                or os.getenv("OZON_UPOLOAD_FOLDER")
                or "/data/uploads"
            ),
            tmp_upload_folder=os.getenv("OZON_ENV_TMP_UPLOAD_FOLDER", "/tmp"),
            backend_interface=os.getenv("OZON_BACKEND_INTERFACE", "db"),
        )

    def normalized_api_prefix(self) -> str:
        return "/" + str(self.api_prefix or "/v2").strip("/")

    def normalized_auth_mode(self) -> str:
        mode = str(self.auth_mode or "session").strip().lower()
        if mode in {"m2m", "keycloak_m2m"}:
            return "keycloak"
        if mode in {"none", "disabled"}:
            return "none"
        if mode not in {"session", "keycloak"}:
            return "session"
        return mode

    def keycloak_jwks_url(self) -> str:
        if self.keycloak_jwks_url_value:
            return self.keycloak_jwks_url_value
        if not self.oauth_url:
            return ""
        return self.oauth_url.rstrip("/").removesuffix("/token") + "/certs"

    def keycloak_issuer(self) -> str:
        if self.keycloak_issuer_value:
            return self.keycloak_issuer_value
        if not self.oauth_url:
            return ""
        marker = "/protocol/openid-connect/token"
        if marker in self.oauth_url:
            return self.oauth_url.split(marker, maxsplit=1)[0]
        return ""

    def keycloak_algorithms_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.keycloak_algorithms.split(",")
            if item.strip()
        ]

    def ozon_env_cfg(self) -> dict:
        cfg = {
            "app_code": self.app_code,
            "mongo_user": self.mongo_user,
            "mongo_pass": self.mongo_pass,
            "mongo_url": self.mongo_url,
            "mongo_db": self.mongo_db,
            "mongo_replica": self.mongo_replica,
            "models_folder": self.models_folder,
            "backend_interface": "db",
        }
        return {key: value for key, value in cfg.items() if value is not None}
