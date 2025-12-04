import asyncio
import copy
import json
import locale
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Union, List, Dict

from bson import ObjectId, Decimal128
from pydantic._internal._model_construction import ModelMetaclass
from pymongo import ReturnDocument
from pymongo.errors import (
    DuplicateKeyError,
    ConnectionFailure,
    WriteConcernError,
)

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
from ozonenv.core.i18n import _
from ozonenv.core.utils import is_json, traverse_and_convertd_datetime

# --- Constants e Regex (Definite all'esterno o come attributi di classe) ---
ISO_DATETIME_REGEX = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
NAME_ALLOWED_REGEX = re.compile(r"^[A-Za-z0-9._~():+-]*$")
SORT_DIR = {"asc": 1, "desc": -1}

logger = logging.getLogger(__name__)


class OzonMBase:
    def __init__(
        self,
        model_name: str,
        setting_app: Settings = None,
        data_model: str = "",
        session_model: bool = False,
        virtual: bool = False,
        static: BasicModel = None,
        schema: dict = None,
    ):

        if schema is None:
            schema = {}

        # --- Proprietà Base ---
        self.name = model_name
        self.setting_app: Settings = setting_app
        self.virtual = virtual
        self.static: BasicModel = static
        self.session_model = (
            session_model  # Rimosso self.is_session_model duplicato
        )

        # --- Nomi e Schema ---
        self.data_model = data_model if virtual else (data_model or model_name)
        self.schema = copy.deepcopy(schema)

        # --- Utility e Valori di Default ---
        self.sort_dir = SORT_DIR
        self.default_sort_str = "list_order:desc,"
        self.default_domain = {"active": True, "deleted": 0}
        self.archived_domain = {"active": False, "deleted": {"$gt": 0}}
        self.tz = self.setting_app.tz
        self.dte = DateEngine(TZ=self.tz)

        # --- Stato e Configurazione ---
        self.status: BasicReturn = BasicReturn(fail=False, msg="", data={})
        self.name_allowed = NAME_ALLOWED_REGEX
        self.transform_config = {}
        self.virtual_fields_parser = {}
        self.tranform_data_value = {}
        self.queryformeditable = {}

        # --- Schema Properties (Inizializzazione) ---
        self.rheader = False
        self.rfooter = False
        self.send_mail_create = False
        # Rimosso self.send_mail_create duplicato
        self.form_disabled = False
        self.no_submit = False
        self.init_schema_properties()
        self.depends = []
        self.it_depends = []

        # --- Oggetti Modello e Servizio (Assegnati in init_model) ---
        self.model: BasicModel = static
        self.instance: BasicModel
        self.model_meta: ModelMetaclass = None
        self.modelr: CoreModel = None
        self.mm: ModelMaker = None
        self.service: ModelService = None
        self.def_recompute_dv = True
        self.def_recompute_dt = True

    def init_schema_properties(self):
        """Popola gli attributi della classe in base alle proprietà definite nello schema."""
        props = self.schema.get("properties", {})
        if not props:
            return

        for k, v in props.items():
            if k == "sort":
                self.default_sort_str = v
            elif k in [
                "send_mail_create",
                "send_mail_update",
                "rfooter",
                "rheader",
                "form_disabled",
                "no_submit",
            ]:
                setattr(self, k, v == "1")
            elif k == "queryformeditable":
                self.queryformeditable = is_json(v)

    async def init_model(self):
        """Inizializza ModelMaker e ModelService."""
        self.mm = ModelMaker(self.name, tz=self.setting_app.tz)

        if self.static:
            self.model = self.static
            self.tranform_data_value = self.model.tranform_data_value()
            self.depends = self.model.model_depends()
        elif not self.virtual:
            # Crea il modello dinamico da schema se non statico e non virtuale
            c_maker = ModelMaker("component", tz=self.setting_app.tz)
            # Presume che 'Component' sia una classe disponibile, es. Component = BasicModel
            c_maker.model = Component
            c_maker.new()
            self.mm.from_formio(self.schema)
            self.model = self.mm.model  # Assegna il modello creato

        # Inizializza il servizio se il modello è pronto
        if self.model:
            self.service = ModelService(self.model, self.orm, self.tz)

    @classmethod
    def _value_type(cls, v: Any) -> type:
        """Determina il tipo di valore in modo più efficiente."""
        if isinstance(v, (dict, list)):
            return type(v)

        if isinstance(v, str) and v.lower() in ["true", "false"]:
            return bool

        s = str(v)

        # Int check
        try:
            int(s)
            return int
        except ValueError:
            pass

        # Float check
        try:
            float(s)
            return float
        except ValueError:
            pass

        # Datetime check
        if len(s) > 9 and ISO_DATETIME_REGEX.search(s):
            return datetime

        return str

    def parse_data_value(self, val: Any, cfg: dict) -> Any:
        """Trasforma un valore di dato secondo la configurazione (per data_value)."""
        if val is None:
            return None

        if cfg["type"] == 'int':
            return int(val)
        elif cfg["type"] == 'str':
            return str(val)
        elif cfg["type"] == 'datetime':
            return self.dte.format_in_client_tz(val, dt_type="datetime")
        elif cfg["type"] == 'date':
            return self.dte.format_in_client_tz(val, dt_type="date")
        elif cfg["type"] == 'float':
            return self.readable_float(val, dp=cfg.get("dp", 2))
        else:
            return val

    def readable_float(self, val: Any, dp: int = 2, g: bool = True) -> str:
        """Formatta un float per la visualizzazione."""
        if isinstance(val, (str, Decimal128)):
            try:
                val = float(str(val))
            except (ValueError, TypeError):
                return str(val)

        try:
            return locale.format_string(f"%.{dp}f", val, g)
        except Exception:
            return f"{val:.{dp}f}"

    def _make_from_dict(
        self, dict_data: dict, data_value: dict = None
    ) -> dict:
        """
        Metodo ricorsivo per creare un modello da dizionario (principalmente per virtual model).
        """
        res_dict = {}
        if data_value is None:
            data_value = {}

        res_dict["data_value"] = dict_data.get("data_value", {}).copy()

        for k, v in dict_data.items():
            if k == "data_value":
                continue

            if isinstance(v, dict):
                res_dict[k] = self._make_from_dict(v, data_value.get(k))
            elif isinstance(v, list):
                res_dict[k] = [
                    (
                        self._make_from_dict(i, data_value.get(k))
                        if isinstance(i, dict)
                        else i
                    )
                    for i in v
                ]
            else:
                val = v

                # 1. Conversione basata su configurazione (più veloce)
                if k in self.tranform_data_value:
                    cfg = self.tranform_data_value[k]
                    if cfg['type'] == "datetime":
                        val = self.dte.parse_to_utc_datetime(v)
                    elif cfg['type'] == "date":
                        val = self.dte.parse_to_utc_date(v)

                    res_dict["data_value"][k] = self.parse_data_value(val, cfg)

                # 2. Conversione basata su inferenza del tipo (se non configurata)
                elif k not in res_dict["data_value"]:
                    inferred_type = self._value_type(v)

                    if inferred_type is datetime:
                        val = self.dte.parse_to_utc_datetime(v)
                        res_dict["data_value"][k] = self.dte.to_ui(
                            val, "datetime"
                        )
                    elif inferred_type is float:
                        res_dict["data_value"][k] = self.readable_float(v, 2)
                    elif k in data_value:
                        res_dict["data_value"][k] = data_value[k]
                    else:
                        res_dict["data_value"][k] = v

                res_dict[k] = val  # Assegna il valore convertito/originale

        return res_dict

    async def make_data_value(
        self, dati: dict, pdata_value: dict = None
    ) -> dict:
        """Centralizza la logica di normalizzazione datetime e data_value."""
        # Presume che self.service gestisca l'ottimizzazione
        if self.service:
            return await self.service.compute_data_value(dati, pdata_value)
        return dati

    async def _load_data(
        self,
        model: BasicModel,
        data: dict,
        virtual: bool,
        data_model: str,
        is_session_model: bool,
        tz: Any,
        virtual_fields_parser: dict,
        as_virtual: bool = False,
        recompute_dv=True,
        recompute_dt=True,
    ) -> tuple[CoreModel, ModelMaker]:

        mm = None
        data = data.copy()

        if not virtual and not as_virtual:
            # Modello Statico/Reale: Rimuovi la chiamata a normalize_datetime_fields se CoreModel
            # lo fa già all'istanziazione, altrimenti CoreModel deve avere questo metodo.
            if recompute_dt:
                data = model.normalize_datetime_fields(tz, data)
            if recompute_dv:
                data = await self.make_data_value(
                    data, pdata_value=data.get("data_value", {})
                )
            modelr = model(**data)
        else:
            # Modello Virtuale
            if data.get("id"):
                data["id"] = str(data["id"])

            mm = ModelMaker(
                data_model, fields_parser=virtual_fields_parser, tz=tz
            )
            data = self._make_from_dict(data)  # Trasforma e popola data_value

            mm.from_data_dict(data)
            modelr = mm.new()

        if not is_session_model and not modelr.rec_name:
            modelr.rec_name = f"{data_model}.{modelr.id}"

        return modelr, mm

    async def load_data(
        self,
        data: dict,
        as_virtual: bool = False,
        recompute_dv=True,
        recompute_dt=True,
    ):
        """Carica i dati in un'istanza del modello (self.modelr)."""
        if self.transform_config:
            self.tranform_data_value = self.transform_config

        self.modelr, self.mm = await self._load_data(
            self.model,
            data,
            self.virtual,
            self.data_model,
            self.session_model,
            self.tz,
            self.virtual_fields_parser,
            as_virtual=as_virtual,
            recompute_dv=recompute_dv,
            recompute_dt=recompute_dt,
        )


class OzonModelBase(OzonMBase):

    # --- Proprietà Semplificate ---
    @property
    def message(self) -> str:
        return self.status.msg

    @property
    def unique_fields(self) -> List[str]:
        return self.model.unique_fields

    @property
    def form_fields(self) -> List[str]:
        return self.model.all_fields()

    @property
    def table_columns(self) -> List[str]:
        return self.model.table_columns()

    # --- Metodi di Stato e Utility ---
    def error_status(self, msg: str, data: dict):
        self.status.fail = True
        self.status.msg = msg
        self.status.data = data

    def init_status(self):
        self.status.fail = False
        self.status.msg = ""
        self.status.data = {}

    def is_error(self) -> bool:
        return self.status.fail

    def chk_write_permission(self) -> bool:
        return True

    async def _pre_execute_check(
        self, domain: dict = None, require_write: bool = False
    ) -> bool:
        """Centralizza il controllo dello stato e della validità del modello virtuale."""
        self.init_status()

        if require_write and not self.chk_write_permission():
            msg = _("Session is Readonly")
            self.error_status(msg, data=domain or {})
            return False

        if self.virtual and not self.data_model:
            msg = _(
                "Data Model is required for virtual model to get data from db"
            )
            self.error_status(msg, data=domain or {})
            return False

        return True

    def get_domain(self, domain: dict = None) -> dict:
        _domain = self.default_domain.copy()
        _domain.update(domain or {})
        return _domain

    def get_domain_archived(self, domain: dict = None) -> dict:
        _domain = self.archived_domain.copy()
        _domain.update(domain or {})
        return _domain

    async def set_lang(self):
        self.lang = self.orm.lang

    def eval_sort_str(self, sortstr: str = "") -> Dict[str, int]:
        """Valuta la stringa di ordinamento nel formato 'campo:direzione,'."""
        sortstr = sortstr or self.default_sort_str
        sort = {}
        for rule_str in sortstr.split(","):
            if rule_str:
                rule_list = rule_str.split(":")
                if len(rule_list) > 1 and rule_list[1] in self.sort_dir:
                    sort[rule_list[0]] = self.sort_dir[rule_list[1]]
        return sort

    def get_dict(
        self,
        rec: CoreModel,
        exclude: list = None,
        compute_datetime: bool = True,
    ) -> dict:
        if exclude is None:
            exclude = []
        return rec.get_dict(exclude=exclude, compute_datetime=compute_datetime)

    def get_dict_record(
        self,
        rec: CoreModel,
        rec_name: str = "",
        compute_datetime: bool = True,
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

    async def parallel_dump(self, models: list) -> str:
        """Esegue il dump in parallelo di una lista di modelli in formato JSON."""
        max_concurrency = 10
        sem = asyncio.Semaphore(max_concurrency)

        async def worker(model):
            async with sem:
                return await model.dump_model_async()

        tasks = [worker(m) for m in models if m]
        json_list = await asyncio.gather(*tasks)
        return "[" + ",".join(json_list) + "]"

    # --- Operazioni sul DB (CRUD) ---

    async def init_unique(self):
        for field in self.model.get_unique_fields():
            await self.set_unique(field)

    async def set_unique(self, field_name: str):
        if not await self._pre_execute_check():
            return
        component_coll = self.db.engine.get_collection(self.data_model)
        await component_coll.create_index([(field_name, 1)], unique=True)

    async def count_by_filter(self, domain: dict) -> int:
        if not await self._pre_execute_check(domain):
            return 0
        coll = self.db.engine.get_collection(self.data_model)
        val = await coll.count_documents(domain)
        return int(val) if val else 0

    async def count(self, domain: dict = None) -> int:
        domain = domain or self.default_domain
        return await self.count_by_filter(domain)

    async def by_name(
        self,
        name: str,
        recompute_dv=True,
        recompute_dt=True,
    ) -> Union[None, CoreModel]:
        return await self.load({'rec_name': name})

    async def new(
        self,
        data: dict = None,
        rec_name: str = "",
        data_value: dict = None,
        trnf_config: dict = None,
        fields_parser: dict = None,
    ) -> Union[None, CoreModel]:

        data = data or {}
        trnf_config = trnf_config or {}
        fields_parser = fields_parser or {}
        data_value = data_value or {}

        # if not await self._pre_execute_check(data, require_write=True):
        #     return None

        # 1. Configurazione e Assegnazione Iniziale
        if rec_name:
            if not self.session_model or self.virtual:
                data["rec_name"] = rec_name

        self.virtual_fields_parser = fields_parser
        self.transform_config = trnf_config

        # 2. Pre-elaborazione Dati (Ottimizzato: chiama make_data_value una sola volta)
        if not self.virtual:
            data = self.model.normalize_datetime_fields(self.tz, data)
            data = await self.make_data_value(
                data, pdata_value=data.get("data_value", {})
            )
        else:
            if data_value:
                data.setdefault('data_value', {}).update(data_value)

        # 3. Caricamento e Validazione
        await self.load_data(data)

        if not self.modelr:
            return None

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
        rec_name: str = "",
        data_value: dict = None,
        trnf_config: dict = None,
        fields_parser: dict = None,
    ) -> Union[None, CoreModel]:

        data = data.get_dict() if isinstance(data, CoreModel) else (data or {})
        trnf_config = trnf_config or {}
        fields_parser = fields_parser or {}
        data_value = data_value or {}

        if not await self._pre_execute_check(data, require_write=True):
            return None

        # 1. Gestione rec_name
        if rec_name:
            data["rec_name"] = rec_name

        if not data.get("rec_name"):
            data["rec_name"] = f"{self.name}.{str(uuid.uuid4().hex)}"

        if not self.name_allowed.match(data["rec_name"]):
            msg = _("Not allowed chars in field name: %s") % data["rec_name"]
            self.error_status(msg, data=data)
            return None

        exist = await self.by_name(data["rec_name"])

        # 2. Pre-elaborazione e Caricamento
        self.virtual_fields_parser = fields_parser
        self.transform_config = trnf_config
        if self.virtual and data_value:
            data.setdefault('data_value', {}).update(data_value)

        await self.load_data(data)

        if not self.modelr:
            return None

        if exist:
            return await self.update(self.modelr)
        else:
            return await self.insert(self.modelr)

    async def insert(
        self,
        record: CoreModel,
        is_many: bool = False,
        recompute_dv=True,
        recompute_dt=True,
    ) -> Union[None, CoreModel]:

        if not await self._pre_execute_check(
            record.get_dict_copy(), require_write=True
        ):
            return None

        if not record.rec_name or not self.name_allowed.match(record.rec_name):
            msg = _("Not allowed chars in field name: %s") % record.get(
                "rec_name"
            )
            self.error_status(msg, data=record.get_dict_json())
            return None

        try:
            coll = self.db.engine.get_collection(self.data_model)

            # 1. Assegnazione Metadati
            record.create_datetime = record.utc_now()
            record = self.set_user_data(record, self.user_session)
            if not is_many:
                record.list_order = await self.count()
            record.active = True

            data = record.get_dict(compute_datetime=False)

            # 2. Preparazione per il Salvataggio
            if not self.virtual:
                if recompute_dt:
                    data = record.normalize_datetime_fields(self.tz, data)
                if recompute_dv:
                    to_save = await self.make_data_value(
                        data, pdata_value=data.get("data_value", {})
                    )
            else:
                to_save = self._make_from_dict(
                    data, data_value=data.get("data_value", {})
                )

            if "_id" not in to_save and 'id' in to_save:
                to_save['_id'] = ObjectId(to_save['id'])

            # 3. Salvataggio e Caricamento
            query_for_insertion = {"_id": to_save['_id']}
            result_save = await coll.find_one_and_replace(
                query_for_insertion,  # Query fittizia: non troverà nulla
                to_save,  # Il documento da inserire
                # Opzioni cruciali:
                upsert=True,
                return_document=ReturnDocument.AFTER,  # Restituisce il documento *dopo* l'inserimento
            )
            # result_save = await coll.insert_one(to_save)
            if result_save:
                result_save.pop("_id", None)
                if isinstance(result_save.get("id"), ObjectId):
                    result_save["id"] = str(result_save["id"])
                await self.load_data(result_save)
                # print(self.modelr)
                return self.modelr

            self.error_status(
                _("Error save  %s ") % str(to_save['rec_name']), to_save
            )
            return None

        except DuplicateKeyError as e:
            logger.error(f"Duplicate {e.details['errmsg']}")
            field = e.details["keyValue"]
            key = list(field.keys())[0]
            val = field[key]
            self.error_status(
                _("Duplicate key error %s: %s") % (str(key), str(val)),
                record.get_dict_copy(),
            )
            return None
        except ConnectionFailure as e:
            # Errore di Connessione: Il driver non è riuscito a connettersi a MongoDB
            # (es. server down o problemi di rete).
            msg = _(f"Error connecting to MongoDB: % {e.details['errmsg']}")
            logger.error(msg)
            self.error_status(
                msg,
                record.get_dict_copy(),
            )
            return None
        except WriteConcernError as e:
            # Errore di Write Concern: Problemi legati alla garanzia di scrittura
            # (es. non abbastanza nodi del replica set hanno confermato la scrittura).
            msg = _(f"Error Write Concern: {e.details['errmsg']}")
            logger.error(msg)
            self.error_status(
                msg,
                record.get_dict_copy(),
            )
            return None
        except Exception as e:
            logger.error(f" Error during insert: {e}", exc_info=True)
            self.error_status(
                _("Operation Error %s ") % str(e), record.get_dict_copy()
            )
            return None

    async def _insert(
        self,
        record: CoreModel,
        count: int,
        recompute_dv=True,
        recompute_dt=True,
    ) -> Union[None, CoreModel]:
        record.list_order = count
        return await self.insert(record, is_many=True)

    async def insert_many(
        self, records: List[CoreModel], resp_type: str = "model"
    ) -> Union[List[Union[None, CoreModel]], str, List[dict]]:

        if not records:
            return []

        start = await self.count()
        results = await asyncio.gather(
            *(self._insert(r, start + records.index(r)) for r in records)
        )

        if resp_type == "json":
            return await self.parallel_dump(results)
        elif resp_type == "dict":
            res = await self.parallel_dump(results)
            return json.loads(res)

        return results

    async def copy(self, domain: dict) -> Union[None, CoreModel]:
        if not await self._pre_execute_check(domain, require_write=True):
            return None

        if self.session_model or self.virtual:
            self.error_status(
                _(
                    "Duplicate session instance or virtual model is not allowed"
                ),
                domain,
            )
            return None

        domain = traverse_and_convertd_datetime(domain)
        record_to_copy = await self.load(domain, in_execution=True)

        if not record_to_copy:
            return None

        self.modelr.renew_id()
        name = self.modelr.rec_name
        self.modelr.rec_name = (
            f"{name}_copy"
            if name and self.name not in name
            else f"{self.data_model}.{self.modelr.id}"
        )

        self.modelr.list_order = await self.count()
        self.modelr.create_datetime = self.modelr.utc_now()
        self.modelr.update_datetime = self.modelr.utc_now()

        for k in self.model.get_unique_fields():
            if k not in ["rec_name"] and hasattr(self.modelr, k):
                val = getattr(self.modelr, k)
                setattr(self.modelr, k, f"{val}_copy")

        self.modelr.set_active()
        self.modelr = self.set_user_data(self.modelr, self.user_session)

        return self.modelr

    async def update(
        self,
        record: CoreModel,
        remove_mata: bool = True,
        force_update_whole_record: bool = True,
        recompute_dv=True,
        recompute_dt=True,
    ) -> Union[None, CoreModel]:

        if not await self._pre_execute_check(
            record.get_dict_copy(), require_write=True
        ):
            return None

        try:
            coll = self.db.engine.get_collection(self.data_model)
            original = await self.load(
                record.rec_name_domain(),
                recompute_dt=recompute_dt,
                recompute_dv=recompute_dv,
            )

            if not original:
                self.error_status(
                    _("Record to update not found"), record.rec_name_domain()
                )
                return None

            # 1. Preparazione Dati
            if not self.virtual:
                data = record.get_dict()
                data["update_uid"] = self.orm.user_session.get("user.uid")
                data["update_datetime"] = record.utc_now()
                if recompute_dt:
                    data = self.model.normalize_datetime_fields(self.tz, data)

                _save = await self.make_data_value(
                    data, pdata_value=data.get("data_value", {})
                )

                if not force_update_whole_record:
                    to_set = original.get_dict_diff(
                        _save,
                        ignore_fields=default_list_metadata_fields_update,
                        remove_ignore_fileds=remove_mata,
                    )
                else:
                    to_set = _save.copy()
            else:
                _save = record.get_dict(compute_datetime=False)
                to_set = self._make_from_dict(_save)

            if not to_set:
                return record

            domain = record.rec_name_domain()

            # *******************************************************************
            # CRUCIALE per EVITARE l'errore "ImmutableField" durante l'aggiornamento:
            # Rimuovi l'ID dal payload che andrà in $set / replacement!
            # *******************************************************************
            # del to_set['_id']
            # to_set.pop("rec_name", None)
            to_set.pop("_id", None)

            # 2. Update nel DB
            # await coll.update_one(record.rec_name_domain(), {"$set": to_set})
            result_doc = await coll.find_one_and_replace(
                domain,
                to_set,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            # print(result_doc)
            # 3. Ritorna il record aggiornato
            # return await self.load(
            #     record.rec_name_domain(),
            #     in_execution=True,
            #     recompute_dt=False,
            #     recompute_dv=True,
            # )
            result_doc.pop("_id", None)
            if isinstance(result_doc.get("id"), ObjectId):
                result_doc["id"] = str(result_doc["id"])
            await self.load_data(result_doc)
            # print(self.modelr)
            return self.modelr

        except DuplicateKeyError as e:
            logger.error(f" Duplicate {e.details['errmsg']}")
            field = e.details["keyValue"]
            key = list(field.keys())[0]
            val = field[key]
            self.error_status(
                _("Duplicate key error %s: %s") % (str(key), str(val)),
                record.get_dict_copy(),
            )
            return None
        except Exception as e:
            logger.error(f" Operation Error: {e}", exc_info=True)
            self.error_status(
                _("Operation Error %s ") % str(e), record.get_dict_copy()
            )
            return None

    async def update_many(
        self,
        records: List[CoreModel],
        force_update_whole_record: bool = True,
        resp_type: str = "model",
    ) -> Union[List[Union[None, CoreModel]], str, List[dict]]:

        results = await asyncio.gather(
            *(
                self.update(
                    r, force_update_whole_record=force_update_whole_record
                )
                for r in records
            )
        )

        if resp_type == "json":
            return await self.parallel_dump(results)
        elif resp_type == "dict":
            res = await self.parallel_dump(results)
            return json.loads(res)

        return results

    async def remove(self, record: CoreModel) -> bool:
        if not await self._pre_execute_check(
            record.get_dict_copy(), require_write=True
        ):
            return False

        coll = self.db.engine.get_collection(self.data_model)
        result = await coll.delete_one(record.rec_name_domain())
        return result.deleted_count > 0

    async def remove_all(self, domain: dict) -> int:
        if not await self._pre_execute_check(domain, require_write=True):
            return 0

        domain = traverse_and_convertd_datetime(domain)
        coll = self.db.engine.get_collection(self.data_model)
        result = await coll.delete_many(domain)
        return result.deleted_count

    async def load(
        self,
        domain: dict,
        in_execution: bool = False,
        recompute_dv=True,
        recompute_dt=True,
    ) -> Union[None, CoreModel]:
        data = await self.load_raw(domain)
        if self.status.fail or not data:
            return None

        await self.load_data(
            data, recompute_dv=recompute_dv, recompute_dt=recompute_dt
        )
        return self.modelr

    async def load_raw(self, domain: dict) -> Union[None, dict]:
        if not await self._pre_execute_check(domain):
            return None

        domain = traverse_and_convertd_datetime(domain)
        coll = self.db.engine.get_collection(self.data_model)
        data = await coll.find_one(domain)

        if not data:
            self.error_status(_("Not found"), domain)
            return None

        data.pop("_id", None)
        if isinstance(data.get("id"), ObjectId):
            data["id"] = str(data["id"])

        return data

    async def process_all(
        self, datas: list, recompute_dv=True, recompute_dt=True
    ) -> List[Any]:
        """Converte i dati raw (da DB) in istanze di modello in parallelo."""
        if not datas:
            return []

        async def process_one(
            rec_data, precompute_dv=True, precompute_dt=True
        ):
            rec_data.pop("_id", None)
            if isinstance(rec_data.get("id"), ObjectId):
                rec_data["id"] = str(rec_data["id"])

            modelr, _ = await self._load_data(
                self.model,
                rec_data,
                self.virtual,
                self.data_model,
                self.session_model,
                self.tz,
                self.virtual_fields_parser,
                recompute_dv=precompute_dv,
                recompute_dt=precompute_dt,
            )
            return modelr

        return await asyncio.gather(
            *(
                process_one(
                    d, precompute_dv=recompute_dv, precompute_dt=recompute_dt
                )
                for d in datas
            )
        )

    async def find(
        self,
        domain: dict,
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        pipeline_items: list = None,
        resp_type: str = "model",
        recompute_dv=False,
        recompute_dt=False,
    ) -> Union[List[Any], str]:

        datas = await self.find_raw(
            domain,
            sort=sort,
            limit=limit,
            skip=skip,
            pipeline_items=pipeline_items or [],
            fields={},
        )

        results = []
        if not self.virtual:
            results = await self.process_all(
                datas, recompute_dv=recompute_dv, recompute_dt=recompute_dt
            )
        else:
            for rec in datas:
                await self.load_data(rec)
                results.append(self.modelr)

        if resp_type == "json":
            return await self.parallel_dump(results)
        elif resp_type == "dict":
            res = await self.parallel_dump(results)
            return json.loads(res)

        return results

    async def find_raw(
        self,
        domain: dict,
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        pipeline_items: list = None,
        fields: dict = None,
    ) -> List[Any]:
        if not await self._pre_execute_check(domain):
            return []

        domain = traverse_and_convertd_datetime(domain)
        _sort = self.eval_sort_str(sort)
        coll = self.db.engine.get_collection(self.data_model)
        pipeline_items = pipeline_items or []
        fields = fields or {}

        if fields and not pipeline_items:
            cursor = coll.find(domain, projection=fields)
            if _sort:
                cursor = cursor.sort(_sort)
            if skip > 0:
                cursor = cursor.skip(skip)
            if limit > 0:
                cursor = cursor.limit(limit)

            return await cursor.to_list(length=None)
        else:
            pipeline = [{"$match": domain}]
            pipeline.extend(pipeline_items)

            if fields:
                pipeline.append({"$project": fields})

            if _sort:
                pipeline.append({"$sort": _sort})

            if skip > 0:
                pipeline.append({"$skip": skip})
            if limit > 0:
                pipeline.append({"$limit": limit})

            datas = coll.aggregate(pipeline)
            return await datas.to_list(length=None)

    async def aggregate_raw(
        self, pipeline: list, sort: str = "", limit: int = 0, skip: int = 0
    ) -> List[Any]:
        if not await self._pre_execute_check():
            return []

        pipeline = traverse_and_convertd_datetime(pipeline)

        if sort:
            _sort = self.eval_sort_str(sort)
            pipeline.append({"$sort": _sort})
        if skip > 0:
            pipeline.append({"$skip": skip})
        if limit > 0:
            pipeline.append({"$limit": limit})

        coll = self.db.engine.get_collection(self.data_model)
        return await coll.aggregate(pipeline).to_list(length=None)

    async def aggregate(
        self,
        pipeline: list,
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        as_virtual: bool = True,
        resp_type: str = "model",
        recompute_dv=False,
        recompute_dt=False,
    ) -> List[Any]:

        datas = await self.aggregate_raw(
            pipeline, sort=sort, limit=limit, skip=skip
        )
        results = []

        if not self.virtual and not as_virtual:
            results = await self.process_all(
                datas, recompute_dv=recompute_dv, recompute_dt=recompute_dt
            )
        else:
            for rec in datas:
                await self.load_data(rec, as_virtual=as_virtual)
                results.append(self.modelr)

        if resp_type == "json":
            return await self.parallel_dump(results)
        elif resp_type == "dict":
            res = await self.parallel_dump(results)
            return json.loads(res)

        return results

    async def distinct(self, field_name: str, query: dict) -> List[Any]:
        if not await self._pre_execute_check(query):
            return []

        query = traverse_and_convertd_datetime(query)
        coll = self.db.engine.get_collection(self.data_model)
        return await coll.distinct(field_name, query)

    async def search_all_distinct(
        self,
        distinct: str = "",
        query: dict = None,
        compute_label: str = "",
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        raw_result: bool = False,
        recompute_dv=False,
        recompute_dt=False,
    ) -> List[Any]:
        if not await self._pre_execute_check(query):
            return []

        query = traverse_and_convertd_datetime(query or {"deleted": 0})

        # Logica di aggregazione per l'unicità
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
                pipeline,
                sort=sort,
                limit=limit,
                skip=skip,
                recompute_dv=recompute_dv,
                recompute_dt=recompute_dt,
            )

    async def set_to_delete(self, record: CoreModel) -> Union[None, CoreModel]:
        if not await self._pre_execute_check(
            record.get_dict_json(), require_write=True
        ):
            return None

        if self.virtual:
            self.error_status(
                _("Unable to set to delete a virtual model"),
                record.get_dict_copy(),
            )
            return None

        delete_at_datetime = datetime.now() + timedelta(
            days=self.setting_app.delete_record_after_days
        )
        record.set_to_delete(delete_at_datetime.timestamp())
        return await self.update(record)

    async def set_active(self, record: CoreModel) -> Union[None, CoreModel]:
        if not await self._pre_execute_check(
            record.get_dict_json(), require_write=True
        ):
            return None

        if self.virtual:
            self.error_status(
                _("Unable to set to delete a virtual model"),
                record.get_dict_copy(),
            )
            return None

        record.set_active()
        return await self.update(record)
