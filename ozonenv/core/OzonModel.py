import asyncio
import copy
import locale
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Union

import bson
import pydantic
import pymongo
from bson import ObjectId
from pydantic._internal._model_construction import ModelMetaclass
from pymongo.errors import DuplicateKeyError, OperationFailure

from ozonenv.core.BaseModels import (
    Component,
    BasicModel,
    CoreModel,
    Settings,
    BasicReturn,
    DictRecord,
    default_list_metadata,
    default_list_metadata_fields_update,
)
from ozonenv.core.DateEngine import DateEngine
from ozonenv.core.ModelMaker import ModelMaker
from ozonenv.core.ModelService import ModelService
from ozonenv.core.exceptions import SessionException
from ozonenv.core.i18n import _
from ozonenv.core.utils import is_json

logger = logging.getLogger(__name__)


class OzonMBase:
    def __init__(
        self,
        model_name,
        setting_app: Settings = None,
        data_model: str = "",
        session_model: BasicModel = False,
        virtual=False,
        static: BasicModel = None,
        schema: dict = None,
    ):
        """

        :param model_name: the name of model must be unique
        :param setting_app: base App settings
        :param data_model: the name of data model in case of virtual model
                           use this collection to store o retreive data.
        :param session_model: True/False if the model is Session or
                              a subclass of Session Model
        :param virtual: True/False if is virtual_model create a model from a
                        generic data dictionary, without the schema
        :param static: ModelClass, if the model is in python Class you need to
                    set model as a static model, when object init the data
                    model, use directly this model class insted to run model
                    maker.
        :param schema: formio form schema, mandatory if
        """

        if schema is None:
            schema = {}
        self.model = None
        self.name = model_name
        self.setting_app: Settings = setting_app
        self.virtual = virtual
        self.static: BasicModel = static
        self.instance: BasicModel
        if self.virtual:
            self.data_model = data_model
        else:
            self.data_model = data_model or model_name
        self.schema = copy.deepcopy(schema)
        self.session_model = session_model
        self.is_session_model = session_model
        self.model_meta: ModelMetaclass = None
        self.modelr: CoreModel = None
        self.mm: ModelMaker = None
        self.model: BasicModel
        self.name_allowed = re.compile(r"^[A-Za-z0-9._~():+-]*$")
        self.sort_dir = {"asc": 1, "desc": -1}
        self.default_sort_str = "list_order:desc,"
        self.default_domain = {"active": True, "deleted": 0}
        self.archived_domain = {"active": False, "deleted": {"$gt": 0}}
        self.transform_config = {}
        self.virtual_fields_parser = {}
        self.status: BasicReturn = BasicReturn(
            **{"fail": False, "msg": "", "data": {}}
        )
        self.tranform_data_value = {}
        self.rheader = False
        self.rfooter = False
        self.send_mail_create = False
        self.send_mail_create = False
        self.form_disabled = False
        self.no_submit = False
        self.queryformeditable = {}
        self.tz = self.setting_app.tz
        self.dte = DateEngine(TZ=self.tz)
        self.init_schema_properties()
        self.depends = []
        self.it_depends = []
        self.service: ModelService

    def init_schema_properties(self):
        if self.schema.get("properties", {}):
            for k, v in self.schema.get("properties", {}).items():
                match k:
                    case ["sort"]:
                        self.default_sort_str = v
                    case [
                        "send_mail_create",
                        "send_mail_update",
                        "rfooter",
                        "rheader",
                        "form_disabled",
                        "no_submit",
                    ]:
                        setattr(self, k, v == "1")
                    case ["queryformeditable"]:
                        self.queryformeditable = is_json(v)

    async def init_model(self):
        self.mm = ModelMaker(self.name, tz=self.setting_app.tz)

        if self.static:
            self.model: BasicModel = self.static
            self.tranform_data_value = self.model.tranform_data_value()
            self.depends = self.model.model_depends()
            self.service = ModelService(self.model, self.orm, self.tz)
        elif not self.static and not self.virtual:
            c_maker = ModelMaker("component", tz=self.setting_app.tz)
            c_maker.model = Component
            c_maker.new()
            self.mm.from_formio(self.schema)

    @classmethod
    def _value_type(cls, v):
        type_def = {
            "int": int,
            "string": str,
            "float": float,
            "dict": dict,
            "list": list,
            "date": datetime,
        }

        ISO_DATETIME_REGEX = re.compile(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?"  # frazioni di secondo opzionali
            r"(?:Z|[+-]\d{2}:\d{2})?"  # timezone opzionale
        )
        s = v
        if type(v) in [dict, list]:
            return type(v)

        if not isinstance(v, str):
            s = str(v)

        regex = re.compile(
            r"(?P<dict>\{[^{}]+\})|(?P<list>\[[^]]+\])|(?P<float>\d*\.\d+)"
            r"|(?P<int>\d+)|(?P<string>[a-zA-Z]+)"
        )
        try:
            x = int(s)
            return int
        except Exception as e:
            pass

        try:
            x = float(s)
            return float
        except Exception as e:
            pass
        if len(s) > 9:

            if bool(ISO_DATETIME_REGEX.search(s)):
                return datetime

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

    def parse_data_value(self, val, cfg):
        if not val:
            return val
        if cfg["type"] == 'int':
            res = int(val)
        elif cfg["type"] == 'str':
            res = str(val)
        elif cfg["type"] == 'datetime':
            res = self.dte.format_in_client_tz(val, dt_type="datetime")
        elif cfg["type"] == 'date':
            res = self.dte.format_in_client_tz(val, dt_type="date")
        elif cfg["type"] == 'float':
            res = self.readable_float(val, dp=cfg["dp"])
        else:
            res = val
        return res

    def readable_float(self, val, dp=2, g=True):
        if isinstance(val, str):
            val = float(val)
        return locale.format_string(f"%.{dp}f", val, g)

    def _make_from_dict(self, dict_data, data_value: dict = None):
        res_dict = {}
        if data_value is None:
            data_value = {}
        for k, v in dict_data.items():
            if isinstance(v, dict):  # For DICT
                if not k == "data_value":
                    res_dict[k] = self._make_from_dict(v, data_value)
            elif isinstance(v, list):  # For LIST
                res_dict[k] = []
                for i in v:
                    if isinstance(i, dict):
                        res_dict[k].append(self._make_from_dict(i, data_value))
                    else:
                        res_dict[k].append(i)
            if "data_value" not in res_dict or not isinstance(
                res_dict.get("data_value"), dict
            ):
                res_dict["data_value"] = {}
            if k in self.tranform_data_value:
                if self.tranform_data_value[k]['type'] == "datetime":
                    v = self.dte.parse_to_utc_datetime(v)
                elif self.tranform_data_value[k]['type'] == "date":
                    v = self.dte.parse_to_utc_date(v)
                val = v
                res_dict["data_value"][k] = self.parse_data_value(
                    val, self.tranform_data_value[k]
                )
            elif self._value_type(v) in [datetime, 'datetime']:
                v = self.dte.parse_to_utc_datetime(v)
                res_dict["data_value"][k] = self.dte.to_ui(v, "datetime")
            elif self._value_type(v) in [float, 'float']:
                res_dict["data_value"][k] = self.readable_float(v, 2)
            else:
                if k in data_value:
                    res_dict["data_value"][k] = data_value[k]
                elif k not in res_dict["data_value"]:
                    res_dict["data_value"][k] = v

            res_dict[k] = v

        return res_dict.copy()

    async def make_data_value(
        self, dati: dict, pdata_value: dict = None
    ) -> dict:
        """
        Controlla tutti i campi datetime del model:
          - se il valore è naive, assume che sia in self.tz
          - lo converte in UTC e aggiorna il dizionario
        Ritorna il dizionario modificato
        """
        return await self.service.compute_data_value(dati, pdata_value)

    async def _load_data(
        self,
        model,
        data,
        virtual,
        data_model,
        is_session_model,
        tz,
        virtual_fields_parser,
    ) -> tuple[CoreModel, ModelMaker]:
        mm = False
        if not virtual:
            # if not in_execution:
            data = model.normalize_datetime_fields(tz, data)
            modelr = model(**data)
        else:
            if data.get("id") is ObjectId:
                data["id"] = str(data["id"])
            # data = BasicModel.normalize_datetime_fields(tz, data)
            mm = ModelMaker(
                data_model,
                fields_parser=virtual_fields_parser,
                tz=tz,
            )
            data = self._make_from_dict(copy.deepcopy(data))
            mm.from_data_dict(data)

            modelr = mm.new()
        if not is_session_model and not modelr.rec_name:
            modelr.rec_name = f"{data_model}.{modelr.id}"
        return modelr, mm

    async def load_data(self, data, in_execution=False):
        if self.transform_config:
            self.tranform_data_value = self.transform_config.copy()
        self.modelr, self.mm = await self._load_data(
            self.model,
            data,
            self.virtual,
            self.data_model,
            self.is_session_model,
            self.tz,
            self.virtual_fields_parser,
        )


class OzonModelBase(OzonMBase):
    @property
    def message(self):
        return self.status.msg

    @property
    def unique_fields(self):
        return self.model.unique_fields

    @property
    def form_fields(self):
        return self.model.all_fields()

    @property
    def table_columns(self):
        return self.model.table_columns()

    def error_status(self, msg, data):
        self.status.fail = True
        self.status.msg = msg
        self.status.data = data

    def init_status(self):
        self.status.fail = False
        self.status.msg = ""
        self.status.data = {}

    def chk_write_permission(self) -> bool:
        return True

    def is_error(self):
        return self.status.fail

    def get_domain(self, domain={}):
        _domain = self.default_domain.copy()
        _domain.update(domain)
        return _domain

    def get_domain_archived(self, domain={}):
        _domain = self.archived_domain.copy()
        _domain.update(domain)
        return _domain

    async def set_lang(self):
        self.lang = self.orm.lang

    def eval_sort_str(self, sortstr="") -> dict[Any, Any]:
        """
        eval sort string in sort rule
        :param sortstr: eg. list_order:asc,rec_name:desc
        :return: List(Tuple) eg. [('list_order', 1),('rec_name', -1)]
        """
        if not sortstr:
            sortstr = self.default_sort_str
        sort_rules = sortstr.split(",")
        sort = {}
        for rule_str in sort_rules:
            if rule_str:
                rule_list = rule_str.split(":")
                if len(rule_list) > 1:
                    sort[rule_list[0]] = self.sort_dir[rule_list[1]]
        return sort

    def get_dict(
        self,
        rec: CoreModel,
        exclude: list = None,
        compute_datetime: bool = True,
    ) -> CoreModel:
        if exclude is None:
            exclude = []
        return rec.get_dict(exclude=exclude, compute_datetime=compute_datetime)

    def get_dict_record(
        self, rec: CoreModel, rec_name: str = "", compute_datetime: bool = True
    ) -> DictRecord:
        dictd = self.get_dict(
            rec,
            exclude=default_list_metadata + ["_id"],
            compute_datetime=compute_datetime,
        )
        if rec_name:
            dictd["rec_name"] = rec_name
        dat = DictRecord(
            model="virtual", rec_name=rec_name, data=copy.deepcopy(dictd)
        )
        return dat

    def set_user_data(self, record: CoreModel, user: dict = None) -> CoreModel:
        if user is None:
            user = {}
        record.owner_uid = user.get("user.uid")
        record.owner_name = user.get("user.full_name", "")
        record.owner_mail = user.get("user.mail", "")
        record.owner_sector = user.get("sector", "")
        record.owner_sector_id = user.get("sector_id", 0)
        record.owner_personal_type = user.get("user.tipo_personale", "")
        record.owner_job_title = user.get("user.qualifica", "")
        record.owner_function = user.get("function", "")
        return record

    async def init_unique(self):
        for field in self.model.get_unique_fields():
            await self.set_unique(field)

    async def set_unique(self, field_name):
        self.init_status()
        component_coll = self.db.engine.get_collection(self.data_model)
        await component_coll.create_index([(field_name, 1)], unique=True)

    async def count_by_filter(self, domain: dict) -> int:
        self.init_status()
        coll = self.db.engine.get_collection(self.data_model)
        val = await coll.count_documents(domain)
        if not val:
            val = 0
        return int(val)

    async def count(self, domain: dict = None) -> int:
        if domain is None:
            domain = {}
        self.init_status()
        if not domain:
            domain = self.default_domain
        return await self.count_by_filter(domain)

    async def by_name(self, name: str) -> CoreModel:
        return await self.load({'rec_name': name})

    async def new(
        self,
        data: dict = None,
        rec_name="",
        data_value: dict = None,
        trnf_config: dict = None,
        fields_parser: dict = None,
    ) -> Union[None, CoreModel]:
        """

        :param data: dict data for new record.
        :param rec_name: name value  for new record if not specify in data
                         dict or if you want to customize it
        :param data_value: if set fill the recod data_value
        :param trnf_config: dict with info to make data_value
                            see MockWorker1 Test for an exampel
        :param fields_parser: dict with info for parsing data when is ambigous
                              see MockWorker1 Test for an exampel
        :return: CoreModel
        """
        if trnf_config is None:
            trnf_config = {}
        if data is None:
            data = {}
        if fields_parser is None:
            fields_parser = {}
        if data_value is None:
            data_value = {}
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data={})
            return None
        if not self.chk_write_permission():
            raise SessionException(detail="Session is Readonly")
        if not data and rec_name or rec_name and self.virtual:
            if not self.is_session_model:
                data["rec_name"] = rec_name
        if not self.virtual:
            # data = self.decode_datetime(data)
            data = self.model.normalize_datetime_fields(self.tz, data)
            data = await self.make_data_value(
                copy.deepcopy(data), pdata_value=data.get("data_value", {})
            )
        else:
            if self.transform_config:
                self.tranform_data_value = self.transform_config.copy()
            if data_value:
                if "data_value" not in data:
                    data['data_value'] = {}
                data['data_value'].update(data_value)
        self.virtual_fields_parser = fields_parser.copy()
        self.transform_config = trnf_config.copy()
        await self.load_data(data)
        if not self.name_allowed.match(self.modelr.rec_name):
            msg = (
                _("Not allowed chars in field name: %s") % self.modelr.rec_name
            )
            self.error_status(msg, data=data)
            return None
        self.modelr.set_active()
        return self.modelr

    async def upsert(
        self,
        data: Union[dict, CoreModel] = None,
        rec_name="",
        data_value: dict = None,
        trnf_config: dict = None,
        fields_parser: dict = None,
    ) -> Union[None, CoreModel]:
        """

        :param data: dict data for new record.
        :param rec_name: name value  for new record if not specify in data
                         dict or if you want to customize it
        :param data_value: if set fill the recod data_value
        :param trnf_config: dict with info to make data_value
                            see MockWorker1 Test for an exampel
        :param fields_parser: dict with info for parsing data when is ambigous
                              see MockWorker1 Test for an exampel
        :return: CoreModel
        """
        if trnf_config is None:
            trnf_config = {}
        if data is None:
            data = {}
        elif isinstance(data, CoreModel):
            data = data.get_dict()
        if fields_parser is None:
            fields_parser = {}
        if data_value is None:
            data_value = {}
        if self.virtual and not self.data_model:
            self.error_status(_("Cannot update a virtual object"), data)
            return None
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data={})
            return None
        if not self.chk_write_permission():
            raise SessionException(detail="Session is Readonly")
        if not data and rec_name or rec_name and self.virtual:
            data["rec_name"] = rec_name
        if not data.get("rec_name"):
            data["rec_name"] = f"{self.name}.{str(uuid.uuid4().hex)}"
        if not self.name_allowed.match(data["rec_name"]):
            msg = (
                _("Not allowed chars in field name: %s") % self.modelr.rec_name
            )
            self.error_status(msg, data=data)
            return None

        exist = await self.by_name(data["rec_name"])
        if self.virtual:
            if data_value:
                if "data_value" not in data:
                    data['data_value'] = {}
                data['data_value'].update(data_value)
            self.virtual_fields_parser = fields_parser.copy()
            self.transform_config = trnf_config.copy()
        await self.load_data(data)
        # self.modelr.set_active()
        if exist:
            return await self.update(self.modelr)
        else:
            return await self.insert(self.modelr)

    async def insert(self, record: CoreModel) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data={})
            return None
        if self.virtual and not self.data_model:
            self.error_status(
                _("Cannot save on db a virtual object"),
                record.get_dict_copy(),
            )
            return None
        try:
            if not record.rec_name or not self.name_allowed.match(
                record.rec_name
            ):
                msg = _("Not allowed chars in field name: %s") % record.get(
                    "rec_name"
                )
                self.error_status(msg, data=record.get_dict_json())
                return None

            coll = self.db.engine.get_collection(self.data_model)

            record.create_datetime = record.utc_now()
            record = self.set_user_data(record, self.user_session)
            record.list_order = await self.count()
            record.active = True
            data = record.get_dict(compute_datetime=False)
            if not self.virtual:
                data = record.normalize_datetime_fields(self.tz, data)
                to_save = await self.make_data_value(
                    copy.deepcopy(data), pdata_value=data.get("data_value", {})
                )
            else:
                to_save = self._make_from_dict(
                    copy.deepcopy(data), data_value=data.get("data_value", {})
                )

            if "_id" not in to_save:
                to_save['_id'] = bson.ObjectId(to_save['id'])
            result = None
            result_save = await coll.insert_one(to_save)
            if result_save:
                return await self.load(
                    {"rec_name": to_save['rec_name']}, in_execution=True
                )
            self.error_status(
                _("Error save  %s ") % str(to_save['rec_name']), to_save
            )
            return result

        except pymongo.errors.DuplicateKeyError as e:
            logger.error(f" Duplicate {e.details['errmsg']}")
            field = e.details["keyValue"]
            key = list(field.keys())[0]
            val = field[key]
            self.error_status(
                _("Duplicate key error %s: %s") % (str(key), str(val)),
                record.get_dict_copy(),
            )
            return None
        except pydantic.ValidationError as e:
            logger.error(f" Validation {e}")
            self.error_status(
                _("Validation Error  %s ") % str(e), record.get_dict_copy()
            )
            return None

    async def copy(self, domain) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data={})
            return None
        if self.is_session_model or self.virtual:
            self.error_status(
                _(
                    "Duplicate session instance "
                    "or virtual model is not allowed"
                ),
                domain,
            )
            return None
        record_to_copy = await self.load(domain, in_execution=True)
        self.modelr.renew_id()
        if (
            hasattr(record_to_copy, "rec_name")
            and self.name not in record_to_copy.rec_name
        ):
            self.modelr.rec_name = f"{self.modelr.rec_name}_copy"
        else:
            self.modelr.rec_name = f"{self.data_model}.{self.modelr.id}"
        self.modelr.list_order = await self.count()
        self.modelr.create_datetime = BasicModel.utc_now()
        self.modelr.update_datetime = BasicModel.utc_now()
        record = await self.new(
            data=self.modelr.get_dict(compute_datetime=False)
        )
        record = self.set_user_data(record, self.user_session)
        for k in self.model.get_unique_fields():
            if k not in ["rec_name"]:
                record.set(k, f"{record.get(k)}_copy")
        record.set_active()
        return record

    async def update(
        self, record: CoreModel, remove_mata=True
    ) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self.virtual and not self.data_model:
            self.error_status(
                _("Cannot update a virtual object"), record.get_dict_copy()
            )
            return None
        try:
            coll = self.db.engine.get_collection(self.data_model)
            original = await self.load(record.rec_name_domain())
            if not self.virtual:
                data = record.get_dict()
                data["update_uid"] = self.orm.user_session.get("user.uid")
                data["update_datetime"] = record.utc_now()
                data = self.model.normalize_datetime_fields(self.tz, data)

                _save = await self.make_data_value(
                    copy.deepcopy(data), pdata_value=data.get("data_value", {})
                )

                to_save = original.get_dict_diff(
                    _save.copy(),
                    ignore_fields=default_list_metadata_fields_update,
                    remove_ignore_fileds=remove_mata,
                )
            else:
                _save = record.get_dict(compute_datetime=False)
                to_save = self._make_from_dict(copy.deepcopy(_save))
            if "rec_name" in to_save:
                to_save.pop("rec_name")
            await coll.update_one(record.rec_name_domain(), {"$set": to_save})
            return await self.load(record.rec_name_domain(), in_execution=True)
        except pymongo.errors.DuplicateKeyError as e:
            logger.error(f" Duplicate {e.details['errmsg']}")
            field = e.details["keyValue"]
            key = list(field.keys())[0]
            val = field[key]
            self.error_status(
                _("Duplicate key error %s: %s") % (str(key), str(val)),
                record.get_dict_copy(),
            )
            return None

        except pydantic.ValidationError as e:
            logger.error(f" Validation {e}")
            self.error_status(
                _("Validation Error  %s ") % str(e),
                record.get_dict_copy(),
            )
            return None
        except OperationFailure as e:
            logger.error(f" OperationFailure {e}")
            self.error_status(
                _("OperationFailure Error  %s ") % str(e),
                record.get_dict_copy(),
            )
            if e.code == 112:  # WriteConflict (transazioni)
                logger.error(f"Conflitto di scrittura")
                self.error_status(
                    _("Conflitto di scrittura "),
                    record.get_dict_copy(),
                )
                return None
            else:
                return None

    async def remove(self, record: CoreModel) -> bool:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self.virtual and not self.data_model:
            self.error_status(
                _("Cannot delete a virtual object"), record.get_dict_copy()
            )
            return False
        coll = self.db.engine.get_collection(self.data_model)
        await coll.delete_one(record.rec_name_domain())
        return True

    async def remove_all(self, domain) -> int:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data={})
            return None
        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, domain)
            return 0
        coll = self.db.engine.get_collection(self.data_model)
        num = await coll.delete_many(domain)
        return num

    async def load(
        self, domain: dict, in_execution=False
    ) -> Union[None, CoreModel]:
        data = await self.load_raw(domain)
        if self.status.fail:
            return None
        await self.load_data(data, in_execution=in_execution)
        return self.modelr

    async def load_raw(self, domain: dict) -> Union[None, dict]:
        self.init_status()
        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, data=domain)
            return None
        coll = self.db.engine.get_collection(self.data_model)
        data = await coll.find_one(domain)
        if not data:
            self.error_status(_("Not found"), domain)
            return {}
        if data.get("_id"):
            data.pop("_id")
        if isinstance(data.get("id"), ObjectId):
            data["id"] = str(data["id"])
        return data

    async def process_all(self, datas) -> list[Any]:
        if not datas:
            return []

        async def process_one(rec_data):
            rec_data.pop("_id", None)
            if isinstance(rec_data.get("id"), ObjectId):
                rec_data["id"] = str(rec_data["id"])
            modelr, mm = await self._load_data(
                self.model,
                rec_data,
                self.virtual,
                self.data_model,
                self.is_session_model,
                self.tz,
                self.virtual_fields_parser,
            )
            return modelr

        results = await asyncio.gather(*(process_one(d) for d in datas))
        return results

    async def find(
        self, domain: dict, sort: str = "", limit=0, skip=0, pipeline_items=[]
    ) -> list[Any]:
        datas = await self.find_raw(
            domain,
            sort=sort,
            limit=limit,
            skip=skip,
            pipeline_items=pipeline_items,
            fields={},
        )
        res = []
        if not self.virtual:
            return await self.process_all(datas)
        else:
            for rec in datas:
                await self.load_data(rec)
                res.append(self.modelr)
            return res

    async def find_raw(
        self,
        domain: dict,
        sort: str = "",
        limit=0,
        skip=0,
        pipeline_items=[],
        fields={},
    ) -> list[Any]:
        self.init_status()
        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, domain)
            return []
        _sort = self.eval_sort_str(sort)
        coll = self.db.engine.get_collection(self.data_model)
        if fields and not pipeline_items:
            res = []
            if limit > 0:
                datas = (
                    coll.find(domain, projection=fields)
                    .sort(sort)
                    .skip(skip)
                    .limit(limit)
                )
            elif sort:
                datas = coll.find(domain, projection=fields).sort(sort)
            else:
                datas = coll.find(domain, projection=fields)
            if datas:
                return await datas.to_list(length=None)
        else:
            res = []
            pipeline = [{"$match": domain}]
            for item in pipeline_items:
                pipeline.append(item)
            if _sort:
                pipeline.append({"$sort": _sort})
            if limit > 0:
                pipeline.append({"$skip": skip})
                pipeline.append({"$limit": limit})
            datas = coll.aggregate(pipeline)
            if datas:
                return await datas.to_list(length=None)

        return res

    async def aggregate_raw(
        self, pipeline: list, sort: str = "", limit=0, skip=0
    ) -> list[Any]:
        if sort:
            _sort = self.eval_sort_str(sort)
            pipeline.append({"$sort": _sort})
        if limit > 0:
            pipeline.append({"$skip": skip})
            pipeline.append({"$limit": limit})
        coll = self.db.engine.get_collection(self.data_model)
        datas = await coll.aggregate(pipeline).to_list(length=None)
        return datas

    async def aggregate(
        self, pipeline: list, sort: str = "", limit=0, skip=0
    ) -> list[CoreModel]:
        datas = await self.aggregate_raw(
            pipeline, sort=sort, limit=limit, skip=skip
        )
        res = []
        for rec_data in datas:
            # rec_data = json.loads(
            #     json.dumps(rec_dat, cls=JsonEncoder, ensure_ascii=False)
            # )
            agg_mm = ModelMaker(
                f"{self.data_model}.agg", tz=self.setting_app.tz
            )
            if "_id" in rec_data:
                rec_data.pop("_id")
            agg_mm.from_data_dict(rec_data)
            agg_mm.new(),
            res.append(agg_mm.instance)
        return res

    async def distinct(self, field_name: str, query: dict) -> list[Any]:
        self.init_status()
        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, query)
            return []
        coll = self.db.engine.get_collection(self.data_model)
        datas = await coll.distinct(field_name, query)
        return datas

    async def search_all_distinct(
        self,
        distinct="",
        query: dict = {},
        compute_label="",
        sort: str = "",
        limit=0,
        skip=0,
        raw_result=False,
    ) -> list[Any]:
        self.init_status()
        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, query)
            return []
        if not query:
            query = {"deleted": 0}
        label = {"$first": "$title"}
        label_lst = compute_label.split(",")
        project = {
            distinct: {"$toString": f"${distinct}"},
            "value": {"$toString": f"${distinct}"},
            "type": {"$toString": "$type"},
        }
        if compute_label:
            if len(label_lst) > 0:
                block = []
                for item in label_lst:
                    if len(block) > 0:
                        block.append(" - ")
                    block.append(f"${item}")
                    project.update({item: {"$toString": f"${item}"}})
                label = {"$first": {"$concat": block}}

            else:
                project.update(
                    {label_lst[0]: {"$toString": f"${label_lst[0]}"}}
                )
                label = {"$first": f"${label_lst[0]}"}
        else:
            project.update({"title": 1})
        pipeline = [
            {"$match": query},
            {"$project": project},
            {
                "$group": {
                    "_id": "$_id",
                    f"{distinct}": {"$first": f"${distinct}"},
                    "value": {"$first": f"${distinct}"},
                    "title": label,
                    "label": label,
                    "type": {"$first": "$type"},
                }
            },
        ]
        if raw_result:
            return await self.aggregate_raw(
                pipeline, sort=sort, limit=limit, skip=skip
            )
        else:
            return await self.aggregate(
                pipeline, sort=sort, limit=limit, skip=skip
            )

    async def set_to_delete(self, record: CoreModel) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self.virtual:
            msg = _("Unable to set to delete a virtual model")
            self.error_status(msg, record.get_dict_copy())
            return None
        delete_at_datetime = datetime.now() + timedelta(
            days=self.setting_app.delete_record_after_days
        )
        record.set_to_delete(delete_at_datetime.timestamp())
        return await self.update(record)

    async def set_active(self, record: CoreModel) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self.virtual:
            msg = _("Unable to set to delete a virtual model")
            self.error_status(msg, record.get_dict_copy())
            return None
        record.set_active()
        return await self.update(record)
