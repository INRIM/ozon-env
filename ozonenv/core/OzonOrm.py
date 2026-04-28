import asyncio
import copy
import importlib
import json
import logging
import os
import secrets
import sys
import time as time_
import uuid
from contextvars import ContextVar
from datetime import timedelta
from os.path import dirname, exists
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union

import aiofiles
import httpx
from aiopathlib import AsyncPath
from ozonenv.core.BaseModels import (
    DbViewModel,
    Component,
    Settings,
    AttachmentTrash,
    CoreModel,
    Dict,
    BasicModel,
    JobContext,
    User,
)
from ozonenv.core.ModelService import AttachmentError, ModelService
from ozonenv.core.OzonClient import (
    OzonClient,
    OzonDataApiClient,
    make_json_compatible,
)
from ozonenv.core.OzonModel import OzonModelBase, BasicReturn
from ozonenv.core.auth import (
    KeycloakAuthManager,
    KeycloakAuthSettings,
    TokenExpiredError,
    TokenVerificationError,
)
from ozonenv.core.cache.cache_utils import stop_cache  # , init_cache
from ozonenv.core.db.mongodb_utils import (
    connect_to_mongo,
    close_mongo_connection,
    DbSettings,
    Mongo,
    Collection,
    _DocumentType,
)
from ozonenv.core.exceptions import OzonPermissionError
from ozonenv.core.i18n import _
from ozonenv.core.i18n import update_translation
from ozonenv.core.utils import traverse_and_convertd_datetime
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__file__)

MAIN_CACHE_TIME = 800

base_model_path = dirname(__file__)

C_TEMPLATE_DIR: Path = Path(__file__).parents[0] / "custom_templates"


class OzonEnvBase:
    def __init__(
        self,
        cfg: dict = None,
        upload_folder: str = "",
        cls_model=OzonModelBase,
    ):
        if cfg is None:
            cfg = {}
        self.orm: OzonOrm
        self.db: Mongo = None
        self.ozon_client: OzonClient = None
        if not cfg:
            self.config_system = {
                "app_code": os.getenv("APP_CODE"),
                "mongo_user": os.getenv("MONGO_USER"),
                "mongo_pass": os.getenv("MONGO_PASS"),
                "mongo_url": os.getenv("MONGO_URL"),
                "mongo_db": os.getenv("MONGO_DB"),
                "mongo_replica": os.getenv("MONGO_REPLICA"),
                "models_folder": os.getenv("MODELS_FOLDER", "/models"),
                "keycloak_jwks_url": os.getenv("OZON_KEYCLOAK_JWKS_URL"),
                "keycloak_issuer": os.getenv("OZON_KEYCLOAK_ISSUER"),
                "token_audience": os.getenv("OZON_TOKEN_AUDIENCE"),
                "oauth_url": os.getenv("OZON_OAUTH_URL"),
                "client_id": os.getenv("OZON_CLIENT_ID"),
                "client_secret": os.getenv("OZON_CLIENT_SECRET"),
                "upload_folder": os.getenv(
                    "OZON_UPOLOAD_FOLDER", "/data/uploads"
                ),
            }
        else:
            self.config_system = cfg.copy()
        self.backend_interface = (
            str(
                self.config_system.get(
                    "backend_interface",
                    os.getenv("OZON_BACKEND_INTERFACE", "db"),
                )
            )
            .strip()
            .lower()
        )
        if self.backend_interface not in ["db", "rest"]:
            logger.warning(
                "Invalid backend_interface '%s': fallback to 'db'",
                self.backend_interface,
            )
            self.backend_interface = "db"
        self.db_settings = None
        if self.has_complete_db_config():
            self.db_settings = DbSettings(**self.config_system)
        self.model = ""
        self.models: Dict[str, cls_model] = {}
        self.params = {}
        self.session_is_api = False
        self.user_session: Optional[User] = None
        self.session_token = None
        self.current_job_token = ""
        self.current_job_context: Optional[JobContext] = None
        self.current_token_data: dict[str, Any] = {}
        self.use_cache = False
        self.cache_index = "ozon_env"
        self.redis_url = ""
        self.orm_from_cache: bool = False
        self.upload_folder: str = upload_folder
        self.models_folder: str = self.config_system.get(
            "models_folder", "/models"
        )
        self.is_db_local = True
        self.app_code = self.config_system.get("app_code")
        self.cls_model = cls_model
        self.validate_backend_model_interface()
        self.default_tz = (os.getenv("TZ", "Europe/Rome"),)
        self._local_transaction_var = ContextVar(
            f"undo_{id(self)}", default=None
        )
        self.data_value_mode = (
            str(
                self.config_system.get(
                    "data_value_mode",
                    os.getenv("DATA_VALUE_MODE", "runtime"),
                )
            )
            .strip()
            .lower()
        )
        if self.data_value_mode not in ["runtime", "background"]:
            logger.warning(
                "Invalid data_value_mode '%s': fallback to 'runtime'",
                self.data_value_mode,
            )
            self.data_value_mode = "runtime"
        runtime_only_models = self.config_system.get(
            "data_value_runtime_only_models",
            os.getenv(
                "DATA_VALUE_RUNTIME_ONLY_MODELS",
                "component,settings,user,jobcontext",
            ),
        )
        if runtime_only_models is None:
            runtime_only_models = []
        elif isinstance(runtime_only_models, str):
            runtime_only_models = [
                m.strip().lower()
                for m in runtime_only_models.split(",")
                if m.strip()
            ]
        else:
            runtime_only_models = [
                str(m).strip().lower() for m in runtime_only_models
            ]
        self.data_value_runtime_only_models = set(runtime_only_models)
        bg_hours = self.config_system.get(
            "data_value_bg_default_hours",
            os.getenv("DATA_VALUE_BG_DEFAULT_HOURS", "2"),
        )
        try:
            self.data_value_bg_default_hours = int(bg_hours)
        except (TypeError, ValueError):
            self.data_value_bg_default_hours = 2
            logger.warning(
                "Invalid data_value_bg_default_hours '%s': fallback to 2",
                bg_hours,
            )
        self._user_auth_manager: Optional[KeycloakAuthManager] = None
        self._rest_auth_manager: Optional[KeycloakAuthManager] = None

    def has_complete_db_config(self) -> bool:
        required = ["mongo_user", "mongo_pass", "mongo_url", "mongo_db"]
        return all(self.config_system.get(item) for item in required)

    def get_backend_interface(self) -> str:
        return self.backend_interface

    def get_orm_class(self):
        if self.get_backend_interface() == "rest":
            return OzonOrmRest
        return OzonOrm

    def validate_backend_model_interface(self):
        if not isinstance(self.cls_model, type):
            raise TypeError("cls_model must be a class")
        if not issubclass(self.cls_model, OzonModelBase):
            raise TypeError("cls_model must inherit from OzonModelBase")
        interface_type = getattr(self.cls_model, "interface_type", "db")
        if interface_type != self.get_backend_interface():
            raise ValueError(
                "cls_model interface_type '%s' is not coherent with "
                "backend_interface '%s'"
                % (interface_type, self.get_backend_interface())
            )

    def get_current_token_input(self):
        return (
            self.params.get("current_token")
            or self.params.get("access_token")
            or self.params.get("token")
            or {}
        )

    def get_current_job_token(self) -> str:
        return (
            self.params.get("job_token")
            or self.config_system.get("job_token")
            or os.getenv("OZON_JOB_TOKEN", "")
        )

    def _get_auth_settings(self, prefix: str = "") -> KeycloakAuthSettings:
        return KeycloakAuthSettings.from_config(self.config_system, prefix)

    def get_user_auth_manager(self) -> KeycloakAuthManager:
        if self._user_auth_manager is None:
            self._user_auth_manager = KeycloakAuthManager(
                self._get_auth_settings()
            )
        return self._user_auth_manager

    def get_rest_auth_manager(self) -> KeycloakAuthManager:
        if self._rest_auth_manager is None:
            settings = self._get_auth_settings()
            settings.client_id = (
                self.config_system.get("rest_client_id")
                or self.config_system.get("m2m_client_id")
                or os.getenv("OZON_REST_CLIENT_ID", "")
                or os.getenv("OZON_M2M_CLIENT_ID", "")
                or settings.client_id
            )
            settings.client_secret = (
                self.config_system.get("rest_client_secret")
                or self.config_system.get("m2m_client_secret")
                or os.getenv("OZON_REST_CLIENT_SECRET", "")
                or os.getenv("OZON_M2M_CLIENT_SECRET", "")
                or settings.client_secret
            )
            self._rest_auth_manager = KeycloakAuthManager(settings)
        return self._rest_auth_manager

    def local_transaction_start(self):
        if not self._local_transaction_var.get():
            self._local_transaction_var.set({'trlocal': {}})
        for name, model in self.models.items():
            model._transaction = True

    def get_local_transaction(self):
        return self._local_transaction_var.get()

    def local_transaction_add(self, model, action, rec_name, data):
        undo = self._local_transaction_var.get()
        if undo is None:
            undo = {}
        trl = undo['trlocal']
        if model not in trl:
            trl[model] = []
        snap = {"type": action, "data": data, "rec_name": rec_name}
        trl[model].append(snap)
        undo.update(trl)
        self._local_transaction_var.set(undo)

    def local_transaction_end(self):
        self._local_transaction_var.set(None)
        for name, model in self.models.items():
            model._transaction = False

    async def local_transaction_rollback(self):
        undo = self._local_transaction_var.get()
        if not undo:
            return
        trl = undo['trlocal']
        for model, snaps in trl.items():
            coll = self.db.engine.get_collection(model)
            for snap in reversed(snaps):
                if snap["type"] == "insert":
                    await coll.delete_one({"rec_name": snap["rec_name"]})
                elif snap["type"] == "update":
                    await coll.replace_one(
                        {"rec_name": snap["rec_name"]}, snap['data']
                    )
                elif snap["type"] == "delete":
                    await coll.insert_one(snap['data'])
            # ozn_model = self.models.get(model)

        self.local_transaction_end()

    def build_full_graph(self, name, visited=None):
        """
        Costruisce una mappa strutturata centrata su `name`, con:
          - i `depends` sopra (chi influenza il candidato)
          - gli `it_depends` sotto (chi dipende dal candidato)
        Esempio:
        {
          "C": {
              "depends": {"B": {...}},
              "it_depends": {"D": {...}, "E": {...}}
          }
        }
        """
        if visited is None:
            visited = set()
        if name not in self.models or name in visited:
            return {}

        visited.add(name)
        model = self.models[name]

        node = {name: {}}

        # --- sopra: dipendenze
        depends_map = {}
        for dep in getattr(model, "depends", []):
            depends_map.update(self.build_full_graph(dep, visited))

        # --- sotto: chi dipende da me
        it_depends_map = {}
        for dep in getattr(model, "it_depends", []):
            it_depends_map.update(self.build_full_graph(dep, visited))

        node[name]["depends"] = depends_map if depends_map else {}
        node[name]["it_depends"] = it_depends_map if it_depends_map else {}

        return node

    @classmethod
    async def readfilejson(cls, cfg_file):
        async with aiofiles.open(cfg_file, mode="r") as f:
            data = await f.read()
        return json.loads(data)

    @classmethod
    def get_formatted_metrics(cls, start_time: float, time_division=0):
        if time_division > 0:
            process_time = (time_.monotonic() - start_time) / time_division
        else:
            process_time = time_.monotonic() - start_time
        return "{0:.2f}".format(process_time)

    @classmethod
    def fail_response(cls, err, err_details="", data={}):
        if "err_details" not in data:
            data["err_details"] = err_details
        return BasicReturn(fail=True, msg=err, data=data)

    @classmethod
    def success_response(cls, msg, data={}):
        return BasicReturn(fail=False, msg=msg, data=data)

    @classmethod
    def get_value_for_select_list(cls, list_src, key, label_key="label"):
        for item in list_src:
            if item.get("value") == key:
                return item.get(label_key)
        return ""

    async def set_lang(self, lang="it", update=False):
        self.lang = lang
        await run_in_threadpool(update_translation, lang)
        # locale.setlocale(locale.LC_NUMERIC, locale.locale_alias[lang])
        if update:
            await self.orm.set_lang()

    async def insert_update_component(self, schema):
        """
        :param schema: json dict of component with formio schema
        :return: Component record
        """
        c_model = self.get('component')
        model_name = schema.get("rec_name")
        component = await c_model.load({"rec_name": model_name})
        new_component = await c_model.new(data=schema)
        if not component:
            res = await c_model.insert(new_component)
            await self.orm.add_model(model_name)
        else:
            res = await c_model.update(new_component)
            await self.orm.update_model(schema, component)
        return res

    def get(self, model_name) -> OzonModelBase:
        return self.models.get(model_name)

    def is_data_value_runtime_only_model(self, model_name: str) -> bool:
        return model_name.lower() in self.data_value_runtime_only_models

    def is_data_value_runtime_enabled(self, model_name: str) -> bool:
        if self.is_data_value_runtime_only_model(model_name):
            return True
        return self.data_value_mode == "runtime"

    async def update_data_value_bg(
        self,
        window: Optional[str] = "update_dt",
        hours: int = None,
    ) -> dict:
        if hours is None:
            hours = self.data_value_bg_default_hours
        if not self.models:
            return {}
        tasks = {}
        for model_name, model in self.models.items():
            if self.is_data_value_runtime_only_model(model_name):
                continue
            if not hasattr(model, "update_data_value_bg"):
                continue
            tasks[model_name] = asyncio.create_task(
                model.update_data_value_bg(window=window, hours=hours)
            )
        results = {}
        for model_name, task in tasks.items():
            try:
                results[model_name] = await task
            except Exception as exc:
                logger.exception(
                    "update_data_value_bg failed for model %s",
                    model_name,
                )
                results[model_name] = {
                    "model": model_name,
                    "updated": 0,
                    "scanned": 0,
                    "skipped": False,
                    "error": str(exc),
                }
        return results

    async def updata_data_value_bg(
        self,
        window: Optional[str] = "update_dt",
        hours: int = None,
    ) -> dict:
        return await self.update_data_value_bg(window=window, hours=hours)

    def start_update_data_value_bg(
        self,
        window: Optional[str] = "update_dt",
        hours: int = None,
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.update_data_value_bg(window=window, hours=hours)
        )

    def start_updata_data_value_bg(
        self,
        window: Optional[str] = "update_dt",
        hours: int = None,
    ) -> asyncio.Task:
        return self.start_update_data_value_bg(window=window, hours=hours)

    def get_collection(self, collection) -> Collection[_DocumentType]:
        return self.db.engine.get_collection(collection)

    async def add_schema(self, schema: dict) -> OzonModelBase:
        component_model = self.models.get("component")
        component = await component_model.new(schema)
        component = await component_model.insert(component)
        if not component.data_model:
            await self.orm.add_model(component.rec_name, virtual=False)
            return self.get(component.rec_name)
        else:
            return self.get(component.data_model)

    async def add_model(
        self, model_name, virtual=False, data_model=""
    ) -> OzonModelBase:
        if self.user_session and self.user_session.is_public:
            return None
        if model_name not in self.models:
            await self.orm.add_model(
                model_name, virtual=virtual, data_model=data_model
            )
        return self.get(model_name)

    async def add_static_model(
        self, model_name: str, model_class: BasicModel, private: bool = False
    ) -> OzonModelBase:
        return await self.orm.add_static_model(
            model_name, model_class, private
        )

    async def connect_db(self):
        self.db = await connect_to_mongo(self.db_settings)

    async def close_db(self):
        await close_mongo_connection(self.db)

    async def init_orm(
        self,
        db=None,
        local_model: dict = None,
        local_model_private: list = None,
        components: list[dict] = None,
        job_contexts: list[dict] = None,
        settings: dict = None,
    ):
        if local_model is None:
            local_model = {}
        if local_model_private is None:
            local_model_private = []
        if db:
            self.db = db
            self.is_db_local = False
        elif self.db_settings:
            await self.connect_db()
        else:
            self.is_db_local = False
        await self.set_lang()
        orm_cls = self.get_orm_class()
        self.orm = orm_cls(self, cls_model=self.cls_model)
        if isinstance(self.orm, OzonOrmRest):
            self.orm.load_local_definitions(
                components=components,
                job_contexts=job_contexts,
                settings=settings,
            )
        if local_model:
            for k, v in local_model.items():
                if k not in self.orm.orm_models:
                    self.orm.orm_models.append(k)
                self.orm.orm_static_models_map[k] = v
                if k in local_model_private:
                    self.orm.add_private_model(k)

    async def init_env(
        self,
        db: Mongo = None,
        local_model: dict = None,
        local_model_private: list = None,
        components: list[dict] = None,
        job_contexts: list[dict] = None,
        settings: dict = None,
    ):
        if local_model is None:
            local_model = {}
        if local_model_private is None:
            local_model_private = []
        await self.init_orm(
            db=db,
            local_model=local_model,
            local_model_private=local_model_private,
            components=components,
            job_contexts=job_contexts,
            settings=settings,
        )
        await self.orm.init_models()

    async def get_auth_token(
        self,
        username: str = "testuser",
        password: str = "testpass",
        client_id: str | None = None,
        client_secret: str | None = None,
        oauth_url: str | None = None,
    ) -> str:
        token_url = oauth_url or os.environ["OZON_OAUTH_URL"]

        data = {
            "grant_type": "password",
            "client_id": client_id or os.environ["OZON_CLIENT_ID"],
            "client_secret": client_secret or os.environ["OZON_CLIENT_SECRET"],
            "username": username,
            "password": password,
            "audience": client_id or os.environ["OZON_TOKEN_AUDIENCE"],
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(token_url, data=data)
            response.raise_for_status()
            payload = response.json()

        return payload["access_token"]

    async def make_auth_params(
        self, username="adminuser", password="adminpass", **extra
    ) -> dict:
        params = {
            "current_token": await self.get_auth_token(
                username=username,
                password=password,
            )
        }
        params.update(extra)
        return params

    async def login(
        self,
        username,
        password,
        **extra,
    ):
        await self.init_env()
        self.params = await self.make_auth_params(
            username=username, password=password, **extra
        )
        return await self.session_app()

    async def close_env(self):
        if self.is_db_local:
            await self.close_db()
        if self.use_cache:
            await stop_cache()

    async def create_job_context(
        self,
        client_id: str,
        expire_sec: int = 900,
        job_key: str = "",
        process_instance_key: str = "",
    ) -> JobContext:
        return await self.orm.create_job_context(
            client_id=client_id,
            expire_sec=expire_sec,
            job_key=job_key,
            process_instance_key=process_instance_key,
        )

    async def delete_job_context(self, job_token: str = "") -> bool:
        return await self.orm.delete_job_context(
            job_token or self.current_job_token
        )

    async def validate_job_context(
        self,
        job_token: str = "",
        client_id: str = "",
    ) -> JobContext:
        job_token = job_token or self.current_job_token
        job_context = await self.orm.validate_job_context(
            job_token,
            client_id=client_id,
        )
        self.current_job_token = job_token
        self.current_job_context = job_context
        return job_context

    async def verify_job_context(
        self,
        job_token: str = "",
        client_id: str = "",
    ) -> JobContext:
        return await self.validate_job_context(
            job_token=job_token,
            client_id=client_id,
        )

    async def init_api_job_context(
        self,
        m2m_token: str,
        job_token: str,
    ) -> JobContext:
        job_context = await self.orm.init_job_context_auth(
            m2m_token=m2m_token,
            job_token=job_token,
        )
        self.current_job_token = job_token
        self.current_job_context = job_context
        self.user_session = self.orm.user_session
        return job_context

    async def job_done(self, job_token: str = "") -> bool:
        return await self.delete_job_context(job_token=job_token)

    async def clean_job_contexts(self) -> int:
        return await self.orm.clean_job_contexts()

    async def make_app_session(
        self,
        params: dict,
        use_cache=True,
        cache_idx="ozon_env",
        redis_url="redis://redis_cache",
        db=None,
        local_model={},
        local_model_private: list = None,
        components: list[dict] = None,
        job_contexts: list[dict] = None,
        settings: dict = None,
    ) -> BasicReturn:
        try:
            self.params = copy.deepcopy(params)
            self.use_cache = use_cache
            self.cache_index = cache_idx
            self.redis_url = redis_url
            await self.init_env(
                db=db,
                local_model=local_model,
                local_model_private=local_model_private,
                components=components,
                job_contexts=job_contexts,
                settings=settings,
            )
            res = await self.session_app()
            await self.close_env()
            return res
        except Exception as e:
            logger.exception(e)
            return self.fail_response(str(e))

    async def session_app(self) -> BasicReturn:
        self.session_is_api = self.params.get("session_is_api", False)
        self.current_job_token = self.get_current_job_token()
        await self.orm.init_auth(
            self.get_current_token_input(),
            job_token=self.current_job_token,
        )
        if not self.upload_folder:
            self.upload_folder = self.orm.app_settings.upload_folder
        self.user_session = self.orm.user_session
        if not self.user_session:
            return self.fail_response(_("Token not allowed"))
        return BasicReturn(fail=False, msg="Done", data={})


def compute_model_dependencies(name, model, models):
    deps = getattr(model, "depends", [])
    return [(dep, name) for dep in deps if dep in models]


def propagate_data_model_dependencies(reverse_depends, models):
    """
    Propaga le dipendenze ai figli (data_model chain).
    Se X.depends include Y, allora Y.it_depends e tutti i discendenti di Y.it_depends includono X.
    """

    # Mappa padre -> figli diretti
    children_map = {}
    for name, model in models.items():
        parent = getattr(model, "data_model", None)
        if parent and parent in models:
            children_map.setdefault(parent, []).append(name)

    # Funzione ricorsiva per trovare tutti i discendenti
    def get_all_descendants(parent, visited=None):
        if visited is None:
            visited = set()
        if parent not in children_map:
            return set()
        for child in children_map[parent]:
            if child not in visited:
                visited.add(child)
                visited |= get_all_descendants(child, visited)
        return visited

    # Costruiamo la nuova mappa
    new_reverse = {dep: set(deps) for dep, deps in reverse_depends.items()}

    # Per ogni chiave, propaga anche ai discendenti
    for dep, dependents in reverse_depends.items():
        for descendant in get_all_descendants(dep):
            new_reverse.setdefault(descendant, set()).update(dependents)

    # Converte in liste ordinate per consistenza
    reverse_depends.clear()
    for k, v in new_reverse.items():
        reverse_depends[k] = sorted(v)


class OzonOrm:
    def __init__(self, env: OzonEnvBase, cls_model=OzonModelBase):
        self.env: OzonEnvBase = env
        self.lang = env.lang
        self.db: Mongo = env.db
        self.config_system = env.config_system.copy()
        self.user_session: Optional[User] = None
        self.list_auto_models = []
        self.orm_models = [
            "component",
            "user",
            "jobcontext",
            "attachmenttrash",
            "settings",
        ]
        self.orm_static_models_map = {
            "component": Component,
            "user": User,
            "jobcontext": JobContext,
            "attachmenttrash": AttachmentTrash,
            "settings": Settings,
        }
        self.dependencies = {}
        self.db_models = []
        self.orm_sys_models = ["component", "user", "jobcontext", "settings"]
        self.private_models = ["settings"]
        self.models_path = self.env.models_folder
        self.app_settings: Settings = None
        self.app_code = self.env.app_code
        self.cls_model = cls_model
        self.tz = "Europe/Rome"

    def add_private_model(self, name):
        if name not in self.private_models:
            self.private_models.append(name)

    async def add_static_model(
        self, model_name: str, model_class: BasicModel, private: bool = False
    ) -> OzonModelBase:
        _model_name = model_name.replace(" ", "").strip().lower()
        if _model_name not in self.orm_models:
            self.orm_models.append(_model_name)
        self.orm_static_models_map[_model_name] = model_class
        if _model_name not in self.env.models:
            self.env.models[_model_name] = self.cls_model(
                _model_name,
                self,
                static=model_class,
            )
            await self.env.models[_model_name].init_model()
            await self.env.models[_model_name].init_unique()
        if private:
            self.add_private_model(_model_name)
        return self.env.models[_model_name]

    async def init_db_models(self):
        self.db_models = await self.get_collections_names()
        self.app_settings = await self.init_settings(self.app_code)

    def set_it_depends(self, reverse_depends):
        """Aggiorna i model con la lista it_depends."""
        for dep, dependents in reverse_depends.items():
            self.env.models[dep].it_depends = dependents

    async def build_reverse_dependencies(self):
        models = self.env.models
        tasks = [
            asyncio.to_thread(compute_model_dependencies, name, model, models)
            for name, model in models.items()
        ]
        results = await asyncio.gather(*tasks)

        reverse_depends = {}
        for pairs in results:
            for dep, name in pairs:
                reverse_depends.setdefault(dep, []).append(name)

        propagate_data_model_dependencies(reverse_depends, models)
        self.set_it_depends(reverse_depends)

        return reverse_depends

    async def init_models(self):
        # self.models_path = self.config_system.get("models_folder", "/models")
        await self.init_db_models()
        await AsyncPath(self.models_path).mkdir(parents=True, exist_ok=True)
        await AsyncPath(f"{self.models_path}/__init__.py").touch(exist_ok=True)
        self.list_auto_models = AsyncPath(self.models_path).glob("*.py")
        for main_model in self.orm_models:
            if main_model not in self.env.models:
                await self.make_model(main_model)

        for db_model in self.db_models:
            self.dependencies[db_model] = []
            if db_model not in list(self.env.models.keys()):
                home = AsyncPath(f"{self.models_path}/{db_model}.py")
                if await home.exists():
                    await self.import_module_model(db_model)
                    model = self.orm_static_models_map[db_model]
                    component = await self.env.get("component").load(
                        {
                            '$and': [
                                {"rec_name": db_model},
                                {
                                    'update_datetime': {
                                        '$gt': model.get_version()
                                    }
                                },
                            ]
                        }
                    )
                    if component:
                        await self.update_model(
                            component.get_dict_copy(), component
                        )
                    else:
                        await self.make_model(db_model)
                else:
                    await self.add_model(db_model)
        for neme, model in self.orm_static_models_map.items():
            if neme not in self.env.models:
                await self.make_model(neme)
        await self.build_reverse_dependencies()

    async def get_collections_names(self, query={}):
        if not query:
            query = {"name": {"$regex": r"^(?!system\.)"}}
        collection_names = await self.db.engine.list_collection_names(
            filter=query
        )
        q = {
            "$and": [
                {"active": True},
                {"rec_name": {"$nin": collection_names}},
            ]
        }
        coll = self.db.engine.get_collection('component')
        res = await coll.distinct("rec_name", q)
        collection_names += res
        return collection_names

    async def init_settings(self, app_code):
        logger.info(f"init app: {app_code}")
        query = {"rec_name": app_code}
        coll_settings = self.env.get_collection("settings")
        db_settings = await coll_settings.find_one(query)
        if not db_settings:
            db_settings = {
                "rec_name": app_code or "",
                "upload_folder": "/uploads",
                "tz": "Europe/Rome",
            }
        if db_settings.get("_id"):
            db_settings.pop("_id")
        db_settings = Settings.normalize_datetime_fields(self.tz, db_settings)
        return Settings(
            **db_settings,
            exclude_none=True,
            exclude_unset=True,
            check_fields=False,
        )

    async def create_view(self, dbviewcfg: DbViewModel):
        if (
            not dbviewcfg.force_recreate
            and dbviewcfg.name in self.db.engine.collection
        ):
            return False
        collections = await self.get_collections_names()
        if dbviewcfg.force_recreate and dbviewcfg.name in collections:
            self.db.engine.drop_collection(dbviewcfg.name)
        try:
            await self.db.engine.command(
                {
                    "create": dbviewcfg.name,
                    "viewOn": dbviewcfg.model,
                    "pipeline": dbviewcfg.pipeline,
                }
            )
            return True
        except Exception as e:
            logger.error(f" Error create view {dbviewcfg.name} - {e}")
            return False

    def _normalize_token_input(
        self,
        token_input,
    ) -> tuple[str, str, dict[str, Any]]:
        if isinstance(token_input, dict):
            token_data = copy.deepcopy(token_input)
            access_token = (
                token_data.get("access_token")
                or token_data.get("token")
                or token_data.get("access")
                or ""
            )
            refresh_token = token_data.get("refresh_token", "")
        else:
            access_token = str(token_input or "")
            refresh_token = str(self.env.params.get("refresh_token", "") or "")
            token_data = {"access_token": access_token}
            if refresh_token:
                token_data["refresh_token"] = refresh_token
        access_token = KeycloakAuthManager.strip_bearer_token(access_token)
        if access_token:
            token_data["access_token"] = access_token
        return access_token, refresh_token, token_data

    async def load_user_by_uid(self, uid: str) -> Optional[User]:
        if not uid or not self.db:
            return None
        data = await self.db.engine.get_collection("user").find_one(
            {"uid": uid}
        )
        if not data:
            data = await self.db.engine.get_collection("user").find_one(
                {"rec_name": uid}
            )
        if not data:
            return None
        data.pop("_id", None)
        data = User.normalize_datetime_fields(self.tz, data)
        return User(
            **data,
            exclude_none=True,
            exclude_unset=True,
            check_fields=False,
        )

    async def persist_user_token(
        self,
        uid: str,
        token_data: dict[str, Any],
    ):
        if not uid or not token_data or not self.db:
            return
        await self.db.engine.get_collection("user").update_one(
            {"uid": uid},
            {
                "$set": {
                    "token": copy.deepcopy(token_data),
                    "last_login": BasicModel.utc_now(),
                }
            },
        )

    @classmethod
    def extract_auth_roles(cls, claims: dict[str, Any]) -> list[str]:
        roles = claims.get("realm_access", {}).get("roles", [])
        if not isinstance(roles, list):
            roles = []
        return [str(role).strip() for role in roles if str(role).strip()]

    def build_auth_user(
        self,
        verified: Any,
        user_record: Optional[User] = None,
        default_uid: str = "",
    ) -> User:
        claims = copy.deepcopy(verified.claims)
        uid = verified.user_id or default_uid
        data = user_record.get_dict_copy() if user_record else {}
        roles = self.extract_auth_roles(claims)
        if user_record:
            full_name = (
                getattr(user_record, "full_name", "")
                or " ".join(
                    [
                        str(getattr(user_record, "nome", "") or "").strip(),
                        str(getattr(user_record, "cognome", "") or "").strip(),
                    ]
                ).strip()
            )
        else:
            full_name = claims.get("name", "")
        full_name = full_name or claims.get("preferred_username", "") or uid
        mail = (
            getattr(user_record, "mail", "") if user_record else ""
        ) or claims.get("email", "")
        groups = (
            getattr(user_record, "groups", []) if user_record else []
        ) or claims.get("groups", [])
        if not isinstance(groups, list):
            groups = []
        given_name = (
            getattr(user_record, "nome", "") if user_record else ""
        ) or claims.get("given_name", "")
        family_name = (
            getattr(user_record, "cognome", "") if user_record else ""
        ) or claims.get("family_name", "")
        user = {
            "uid": uid,
            "full_name": full_name,
            "mail": mail,
            "tipo_personale": "",
            "qualifica": "",
        }
        rec_name = getattr(user_record, "rec_name", "") if user_record else uid
        if not user_record:
            data.update(
                {
                    "nome": given_name,
                    "cognome": family_name,
                    "active": True,
                    "use_auth": True,
                    "is_admin": "admin" in roles,
                    "tech_admin": "admin" in roles,
                    "is_public": False,
                    "user_role": roles or ["base"],
                    "user_function": (
                        ("admin" if "admin" in roles else roles[0])
                        if roles
                        else "user"
                    ),
                    "default": True,
                    "demo": False,
                    "tz": self.tz or "Europe/Rome",
                }
            )
        data.update(
            {
                "rec_name": rec_name or uid,
                "uid": uid,
                "full_name": full_name,
                "mail": mail,
                "groups": groups,
                "token": copy.deepcopy(verified.token_data),
                "claims": claims,
                "user": user,
                "client_id": verified.client_id,
            }
        )
        return User(
            **data,
            exclude_none=True,
            exclude_unset=True,
            check_fields=False,
        )

    async def save_user(self, user: User, last_login: Any = None):
        user.create_datetime = BasicModel.utc_now()
        if last_login:
            user.last_login = last_login
        user.list_order = (
            await self.db.engine.get_collection("user").count_documents({})
        ) + 1
        data = user.get_dict_copy()
        data.pop("id", None)
        try:
            await self.db.engine.get_collection("user").insert_one(data)
        except Exception:
            existing_user = await self.load_user_by_uid(user.uid)
            if existing_user:
                return existing_user
            raise
        return await self.load_user_by_uid(user.uid)

    async def provision_auth_user(self, verified: Any) -> Optional[User]:
        if not self.db:
            return self.build_auth_user(verified)
        user = self.build_auth_user(verified)
        return await self.save_user(user, BasicModel.utc_now())

    async def authenticate_user_token(self, token_input) -> Any:
        access_token, refresh_token, token_data = self._normalize_token_input(
            token_input
        )
        if not access_token:
            raise TokenVerificationError("Missing JWT token")
        auth_manager = self.env.get_user_auth_manager()
        try:
            verified = await auth_manager.verify(access_token)
        except TokenExpiredError:
            if not refresh_token:
                claims = auth_manager.decode_unverified(access_token)
                user_record = await self.load_user_by_uid(
                    auth_manager.extract_user_id(claims)
                )
                if user_record and isinstance(user_record.token, dict):
                    refresh_token = user_record.token.get("refresh_token", "")
                    if refresh_token:
                        token_data["refresh_token"] = refresh_token
            if not refresh_token:
                raise
            refreshed = await auth_manager.refresh(refresh_token)
            token_data.update(refreshed)
            access_token = refreshed.get("access_token", "")
            if refreshed.get("refresh_token"):
                token_data["refresh_token"] = refreshed["refresh_token"]
            verified = await auth_manager.verify(access_token)
        verified.access_token = access_token
        verified.refresh_token = token_data.get("refresh_token", "")
        token_data["access_token"] = access_token
        verified.token_data = copy.deepcopy(token_data)
        return verified

    async def init_auth(self, token_input, job_token: str = ""):
        verified = await self.authenticate_user_token(token_input)
        user_record = await self.load_user_by_uid(verified.user_id)
        if not user_record:
            user_record = await self.provision_auth_user(verified)
        if not user_record:
            raise TokenVerificationError("User not found")
        if not user_record.active:
            raise TokenVerificationError("User is inactive")
        self.user_session = self.build_auth_user(verified, user_record)
        self.env.session_token = verified.access_token
        self.env.current_token_data = copy.deepcopy(verified.token_data)
        await self.persist_user_token(
            self.user_session.uid, verified.token_data
        )

    async def load_job_context(self, job_token: str) -> Optional[JobContext]:
        if not job_token or not self.db:
            return None
        data = await self.db.engine.get_collection("jobcontext").find_one(
            {"job_token": job_token}
        )
        if not data:
            return None
        data.pop("_id", None)
        data = JobContext.normalize_datetime_fields(self.tz, data)
        return JobContext(
            **data,
            exclude_none=True,
            exclude_unset=True,
            check_fields=False,
        )

    async def validate_job_context(
        self,
        job_token: str,
        client_id: str,
    ) -> JobContext:
        job_context = await self.load_job_context(job_token)
        if not job_context:
            raise TokenVerificationError("JobContext not found")
        if (
            not job_context.active
            or job_context.expires_at <= BasicModel.utc_now()
        ):
            await self.delete_job_context(job_token)
            raise TokenVerificationError("JobContext expired or inactive")
        if client_id and job_context.client_id != client_id:
            raise TokenVerificationError("JobContext client_id mismatch")
        self.env.current_job_token = job_token
        self.env.current_job_context = job_context
        return job_context

    async def authenticate_rest_token(self, m2m_token: str) -> Any:
        m2m_token = KeycloakAuthManager.strip_bearer_token(m2m_token)
        if not m2m_token:
            raise TokenVerificationError("Missing M2M token")
        auth_manager = self.env.get_rest_auth_manager()
        expected_client_id = auth_manager.settings.client_id
        verified = await auth_manager.verify(
            m2m_token,
            expected_client_id=expected_client_id,
        )
        verified.access_token = m2m_token
        verified.token_data = {"access_token": m2m_token}
        return verified

    async def init_job_context_auth(
        self,
        m2m_token: str,
        job_token: str,
    ) -> JobContext:
        verified = await self.authenticate_rest_token(m2m_token)
        job_context = await self.validate_job_context(
            job_token,
            client_id=verified.client_id,
        )
        user_record = await self.load_user_by_uid(job_context.resolved_user_id)
        if not user_record:
            raise TokenVerificationError("JobContext user not found")
        if not user_record.active:
            raise TokenVerificationError("JobContext user is inactive")
        self.user_session = user_record
        self.env.user_session = user_record
        self.env.session_token = verified.access_token
        self.env.current_token_data = copy.deepcopy(verified.token_data)
        return job_context

    async def create_job_context(
        self,
        client_id: str,
        expire_sec: int = 900,
        job_key: str = "",
        process_instance_key: str = "",
    ) -> JobContext:
        if not self.user_session:
            raise TokenVerificationError("Authenticated user required")
        if not client_id:
            raise ValueError("client_id is required")
        if not self.db:
            raise ValueError(
                "JobContext storage requires a configured database"
            )
        issued_at = BasicModel.utc_now()
        expires_at = issued_at + timedelta(seconds=max(int(expire_sec), 1))
        job_token = f"jctx_{secrets.token_urlsafe(32)}"
        job_context = JobContext(
            rec_name=job_token,
            job_token=job_token,
            client_id=client_id,
            job_key=job_key or str(uuid.uuid4()),
            process_instance_key=process_instance_key or str(uuid.uuid4()),
            resolved_user_id=self.user_session.uid,
            issued_at=issued_at,
            expires_at=expires_at,
            owner_uid=self.user_session.uid,
            owner_name=self.user_session.full_name,
            owner_mail=self.user_session.mail,
            active=True,
        )
        data = job_context.get_dict_copy()
        data.pop("id", None)
        await self.db.engine.get_collection("jobcontext").insert_one(data)
        return job_context

    async def delete_job_context(self, job_token: str) -> bool:
        if not job_token or not self.db:
            return False
        result = await self.db.engine.get_collection("jobcontext").delete_one(
            {"job_token": job_token}
        )
        return result.deleted_count > 0

    async def clean_job_contexts(self) -> int:
        if not self.db:
            return 0
        result = await self.db.engine.get_collection("jobcontext").delete_many(
            {
                "$or": [
                    {"active": False},
                    {"expires_at": {"$lte": BasicModel.utc_now()}},
                ]
            }
        )
        return int(result.deleted_count)

    async def runcmd(self, cmd):
        # for security reason check the command
        if not cmd.startswith("datamodel-codegen --input"):
            return
        res = True
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await proc.communicate()

        logger.info(f"[{cmd!r} exited with {proc.returncode}]")

        return res

    async def import_module_model(self, model_name):

        def smart_title(s):
            # Se il primo carattere è alfabetico, lo metto maiuscolo
            if s and s[0].isalpha():
                return s[0].upper() + s[1:]
            return s

        def camel(snake_str):
            parts = snake_str.split("_")
            return "".join(smart_title(word) for word in parts)

        def _getattribute(obj, name):
            for subpath in name.split("."):
                parent = obj
                obj = getattr(obj, subpath)
            return obj, parent

        mclass = camel(model_name)
        module_name = f"{model_name}"
        file_path = f"{self.models_path}/{model_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        model, parent = _getattribute(module, mclass)
        self.orm_static_models_map[model_name] = model

    async def make_local_model(self, mod, version):
        jdata = mod.mm.model.model_json_schema()
        async with aiofiles.open(
            f"/tmp/{mod.name}.json", "w+", encoding="utf-8"
        ) as mod_file:
            await mod_file.write(json.dumps(jdata, ensure_ascii=False))
        res = await self.runcmd(
            f"datamodel-codegen --input /tmp/{mod.name}.json"
            f" --no-use-union-operator "
            f" --input-file-type jsonschema "
            f" --output {self.models_path}/{mod.name}.py "
            f" --custom-template-dir {C_TEMPLATE_DIR} "
            f" --additional-imports \"ozonenv.core.BaseModels.CoreNestedModel,datetime,zoneinfo\" "
            f" --output-model-type pydantic_v2.BaseModel "
            f" --use-standard-collections "
            f"--base-class ozonenv.core.BaseModels.BasicModel"
        )
        if not res:
            return
        tmp = f"""
    
    @classmethod
    def get_version(cls):
        return '{version}'
        
    @classmethod
    def schema(cls) -> dict:
        return {mod.schema}      
          
    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return {jdata}
                
    @classmethod
    def all_fields(cls) -> list:
        return {mod.mm.fields} 
        
    @classmethod
    def select_fields(cls):
        return {mod.mm.select_fields}

    @classmethod
    def select_options(cls, key: str = None, update_options: dict = None):
        options = {mod.mm.select_options}
        if key and update_options and key in options:
            options[key] = update_options.copy()
        return options.copy()
        
    @classmethod
    def datetime_fields(cls):
        return {mod.mm.datetime_fields}
        
    @classmethod
    def get_unique_fields(cls) -> list:
        return {mod.mm.unique_fields}
        
    @classmethod
    def computed_fields(cls):
        return {mod.mm.computed_fields}
    
    @classmethod
    def no_clone_field_keys(cls):
        return {mod.mm.no_clone_field_keys}
    
    @classmethod
    def tranform_data_value(cls):
        return {mod.mm.tranform_data_value}    
    
    @classmethod
    def fields_limit_value(cls):
        return {mod.mm.fields_limit_value}     
    
    @classmethod
    def create_task_action(cls):
        return {mod.mm.create_task_action}
    
    @classmethod
    def fields_properties(cls):
        return {mod.mm.fields_properties}
        
    @classmethod
    def default_hidden_fields(cls):
        return {mod.mm.default_hidden_fields}

    @classmethod
    def default_readonly_fields(cls):
        return {mod.mm.default_readonly_fields}
    
    @classmethod
    def default_required_fields(cls):
        return {mod.mm.default_required_fields}   
         
    @classmethod
    def filter_keys(cls):
        return {mod.mm.filter_keys}  

    @classmethod
    def table_columns(cls) -> dict:
        return {mod.mm.columns}
        
    @classmethod
    def get_data_model(cls):
        return "{mod.mm.data_model}"
    
    @classmethod
    def conditional(cls) -> dict(str, dict):
        return {mod.mm.conditional}

    @classmethod
    def logic(cls) -> dict(str, dict):
        return {mod.mm.logic}
    
    @classmethod
    def conditional(cls) -> dict(str, dict):
        return {mod.mm.conditional}
        
    @classmethod
    def model_depends(cls):
        return {mod.mm.model_depends}
        
    @classmethod
    def nested_datetime_fields(cls):
        return {mod.mm.nested_datetime_fields}
        
    @classmethod
    def nested_transform_data_value(cls):
        return {mod.mm.nested_transform_data_value}
    
    @classmethod
    def file_fields(cls) -> dict:
        return {mod.mm.file_fields}

"""
        async with aiofiles.open(
            f"{self.models_path}/{mod.name}.py", "a+", encoding="utf-8"
        ) as mod_file:
            await mod_file.write(tmp)

    async def init_model_and_write_code(
        self, model_name, data_model, virtual, schema, component
    ):
        mod = self.cls_model(
            model_name,
            self,
            data_model=data_model,
            static=self.orm_static_models_map.get(model_name, None),
            virtual=virtual,
            schema=schema,
        )
        await mod.init_model()
        await self.make_local_model(mod, component.utc_now().isoformat())

    async def add_model(self, model_name, virtual=False, data_model=""):
        schema = {}
        component = None
        if not virtual:
            component = await self.env.get("component").load(
                {"rec_name": model_name}
            )
            if component:
                schema = component.get_dict_copy()
        if (
            schema
            and model_name not in list(self.orm_static_models_map.keys())
            and not virtual
            and component
        ):
            if not exists(f"{self.models_path}/{model_name}.py"):

                await self.init_model_and_write_code(
                    model_name, data_model, virtual, schema, component
                )

            await self.import_module_model(model_name)
        await self.make_model(
            model_name, schema=schema, virtual=virtual, data_model=data_model
        )
        self.db_models = await self.get_collections_names()

    async def make_model(
        self, model_name, schema: dict = None, virtual=False, data_model=""
    ):
        if schema is None:
            schema = {}
        if model_name in list(self.orm_static_models_map.keys()) or virtual:
            if not data_model and schema:
                data_model = schema.get("data_model", "")
            if (
                not data_model
                and not virtual
                and self.orm_static_models_map[model_name].get_data_model()
            ):
                data_model = self.orm_static_models_map[
                    model_name
                ].get_data_model()
            if data_model and virtual:
                data_model_o = self.env.models.get(data_model)
                if data_model_o and data_model_o.data_model:
                    data_model = data_model_o.data_model
            self.env.models[model_name] = self.cls_model(
                model_name,
                self,
                data_model=data_model,
                static=self.orm_static_models_map.get(model_name, None),
                virtual=virtual,
                schema=schema,
            )
            await self.env.models[model_name].init_model()
            if not virtual:
                if model_name not in self.db_models:
                    await self.env.models[model_name].init_unique()
                    await self.build_reverse_dependencies()

    async def update_model(self, schema, component):
        if schema.get("rec_name") in self.orm_static_models_map:
            self.orm_static_models_map.pop(schema.get("rec_name"))
        await self.init_model_and_write_code(
            schema.get("rec_name"), "", False, schema, component
        )
        await self.import_module_model(schema.get("rec_name"))
        await self.make_model(
            schema.get("rec_name"),
            schema=schema,
            virtual=False,
            data_model="",
        )

    async def set_lang(self):
        self.lang = self.env.lang
        for model_name, model in self.env.models.items():
            await model.set_lang()


def _read_record_value(record: dict, key: str):
    if "." not in key:
        return record.get(key)
    value: Any = record
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _match_local_condition(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "$nin" in expected:
            return value not in expected["$nin"]
        if "$gt" in expected:
            return value is not None and value > expected["$gt"]
        if "$gte" in expected:
            return value is not None and value >= expected["$gte"]
    return value == expected


def _match_local_domain(record: dict, domain: dict) -> bool:
    if not domain:
        return True
    if "$and" in domain:
        return all(
            _match_local_domain(record, item) for item in domain["$and"]
        )
    for key, expected in domain.items():
        if key == "$and":
            continue
        if not _match_local_condition(
            _read_record_value(record, key), expected
        ):
            return False
    return True


class OzonOrmRest(OzonOrm):
    def __init__(self, env: OzonEnvBase, cls_model=OzonModelBase):
        super().__init__(env, cls_model=cls_model)
        self.orm_models = [
            name for name in self.orm_models if name != "jobcontext"
        ]
        self.orm_static_models_map.pop("jobcontext", None)
        self.orm_sys_models = [
            name for name in self.orm_sys_models if name != "jobcontext"
        ]
        self.local_only_models = {"component", "settings"}
        self.local_store = {
            "component": [],
            "settings": [],
            "jobcontext": [],
        }
        self.rest_client = OzonDataApiClient.create(
            base_url=self.config_system.get(
                "rest_base_url",
                os.getenv("OZON_REST_BASE_URL", ""),
            ),
            api_prefix=self.config_system.get(
                "rest_api_prefix",
                os.getenv("OZON_REST_API_PREFIX", "/v2"),
            ),
            token=self.config_system.get(
                "rest_token",
                os.getenv("OZON_REST_TOKEN", ""),
            ),
            job_token=self.env.get_current_job_token(),
            oauth_url=self.config_system.get(
                "rest_oauth_url",
                self.config_system.get(
                    "oauth_url", os.getenv("OZON_OAUTH_URL", "")
                ),
            ),
            oauth_client_id=self.config_system.get(
                "rest_client_id",
                self.config_system.get(
                    "m2m_client_id",
                    os.getenv(
                        "OZON_REST_CLIENT_ID",
                        os.getenv(
                            "OZON_M2M_CLIENT_ID",
                            os.getenv("OZON_CLIENT_ID", ""),
                        ),
                    ),
                ),
            ),
            oauth_client_secret=self.config_system.get(
                "rest_client_secret",
                self.config_system.get(
                    "m2m_client_secret",
                    os.getenv(
                        "OZON_REST_CLIENT_SECRET",
                        os.getenv(
                            "OZON_M2M_CLIENT_SECRET",
                            os.getenv("OZON_CLIENT_SECRET", ""),
                        ),
                    ),
                ),
            ),
            token_audience=self.config_system.get(
                "rest_token_audience",
                self.config_system.get(
                    "token_audience", os.getenv("OZON_TOKEN_AUDIENCE", "")
                ),
            ),
        )

    def is_local_model(self, model_name: str) -> bool:
        return str(model_name).strip().lower() in self.local_only_models

    def get_local_store(self, model_name: str) -> list[dict]:
        name = str(model_name).strip().lower()
        if name not in self.local_store:
            self.local_store[name] = []
        return self.local_store[name]

    def load_local_definitions(
        self,
        components: list[dict] = None,
        job_contexts: list[dict] = None,
        settings: dict = None,
    ):
        if components is None:
            components = self.config_system.get("components", [])
        if settings is None:
            settings = self.config_system.get("settings")
        self.local_store["component"] = [
            copy.deepcopy(item) for item in (components or [])
        ]
        if settings:
            self.local_store["settings"] = [copy.deepcopy(settings)]
        else:
            self.local_store["settings"] = [
                {
                    "rec_name": self.app_code or "",
                    "upload_folder": "/uploads",
                    "tz": "Europe/Rome",
                }
            ]

    @classmethod
    def _extract_api_data(cls, response, default=None):
        if response is None:
            return default
        if isinstance(response, dict):
            for key in ["data", "items", "result", "value"]:
                if key in response:
                    return response[key]
        return response

    async def init_db_models(self):
        self.db_models = await self.get_collections_names()
        self.app_settings = await self.init_settings(self.app_code)
        self.tz = self.app_settings.tz

    async def get_collections_names(self, query={}):
        collection_names = []
        try:
            response = await self.rest_client.get_resource("collections_names")
            remote_names = self._extract_api_data(response, default=[])
            if isinstance(remote_names, list):
                collection_names.extend(
                    [
                        str(name).strip()
                        for name in remote_names
                        if str(name).strip()
                    ]
                )
        except Exception as exc:
            logger.warning("REST collections_names bootstrap failed: %s", exc)
        for item in self.get_local_store("component"):
            model_name = str(item.get("rec_name", "")).strip()
            if model_name and model_name not in collection_names:
                collection_names.append(model_name)
        return collection_names

    async def init_settings(self, app_code):
        query = {"rec_name": app_code}
        local_settings = None
        try:
            response = await self.rest_client.get_resource(
                f"init_settings/{app_code}"
            )
            remote_settings = self._extract_api_data(response, default=None)
            if isinstance(remote_settings, dict) and remote_settings:
                local_settings = copy.deepcopy(remote_settings)
                self.local_store["settings"] = [copy.deepcopy(remote_settings)]
        except Exception as exc:
            logger.warning("REST init_settings bootstrap failed: %s", exc)
        for item in self.get_local_store("settings"):
            if _match_local_domain(item, query):
                local_settings = copy.deepcopy(item)
                break
        if local_settings is None:
            local_settings = {
                "rec_name": app_code or "",
                "upload_folder": "/uploads",
                "tz": "Europe/Rome",
            }
        local_settings = Settings.normalize_datetime_fields(
            self.tz, local_settings
        )
        return Settings(
            **local_settings,
            exclude_none=True,
            exclude_unset=True,
            check_fields=False,
        )

    async def add_static_model(
        self, model_name: str, model_class: BasicModel, private: bool = False
    ) -> OzonModelBase:
        _model_name = model_name.replace(" ", "").strip().lower()
        if _model_name not in self.orm_models:
            self.orm_models.append(_model_name)
        self.orm_static_models_map[_model_name] = model_class
        if _model_name not in self.env.models:
            self.env.models[_model_name] = self.cls_model(
                _model_name,
                self,
                static=model_class,
            )
            await self.env.models[_model_name].init_model()
        if private:
            self.add_private_model(_model_name)
        return self.env.models[_model_name]

    async def init_model_and_write_code_from_schema(self, schema: dict):
        model_name = schema.get("rec_name", "").strip()
        if not model_name:
            return
        mod = self.cls_model(
            model_name,
            self,
            data_model=schema.get("data_model", ""),
            virtual=False,
            schema=schema,
        )
        await mod.init_model()
        await self.make_local_model(mod, BasicModel.utc_now().isoformat())

    async def init_models(self):
        await self.init_db_models()
        await AsyncPath(self.models_path).mkdir(parents=True, exist_ok=True)
        await AsyncPath(f"{self.models_path}/__init__.py").touch(exist_ok=True)

        for main_model in self.orm_models:
            if main_model not in self.env.models:
                await self.make_model(main_model)

        for module_path in sorted(Path(self.models_path).glob("*.py")):
            model_name = module_path.stem
            if model_name == "__init__":
                continue
            if model_name not in self.orm_static_models_map:
                await self.import_module_model(model_name)
            if model_name not in self.env.models:
                await self.make_model(model_name)
            if model_name not in self.dependencies:
                self.dependencies[model_name] = []
            model_class = self.orm_static_models_map.get(model_name)
            if model_class and hasattr(model_class, "schema"):
                schema = model_class.schema()
                if schema:
                    records = self.get_local_store("component")
                    if not any(
                        item.get("rec_name") == model_name for item in records
                    ):
                        records.append(copy.deepcopy(schema))

        for schema in self.get_local_store("component"):
            model_name = schema.get("rec_name", "").strip()
            if not model_name:
                continue
            self.dependencies[model_name] = []
            model_file = f"{self.models_path}/{model_name}.py"
            if model_name not in self.orm_static_models_map:
                if not exists(model_file):
                    await self.init_model_and_write_code_from_schema(schema)
                await self.import_module_model(model_name)
            if model_name not in self.env.models:
                await self.make_model(
                    model_name,
                    schema=schema,
                    virtual=False,
                    data_model=schema.get("data_model", ""),
                )
        await self.build_reverse_dependencies()

    async def init_auth(self, token_input, job_token: str = ""):
        if not job_token:
            raise TokenVerificationError("Missing job_token")
        access_token, _refresh_token, token_data = self._normalize_token_input(
            token_input
        )
        current_user = self.env.params.get("current_user") or {}
        if not isinstance(current_user, dict):
            current_user = {}
        uid = (
            current_user.get("uid")
            or current_user.get("rec_name")
            or f"jobcontext:{job_token}"
        )
        full_name = current_user.get("full_name") or current_user.get(
            "name", ""
        )
        mail = current_user.get("mail") or current_user.get("email", "")
        self.user_session = User(
            rec_name=current_user.get("rec_name") or uid,
            uid=uid,
            full_name=full_name,
            mail=mail,
            is_bot=True,
            token=copy.deepcopy(token_data),
            user={
                "uid": uid,
                "full_name": full_name,
                "mail": mail,
            },
            client_id=self.rest_client.oauth_client_id,
        )
        self.env.session_token = access_token
        self.env.current_token_data = copy.deepcopy(token_data)
        self.env.current_job_token = job_token
        self.rest_client.set_job_token(job_token)

    async def add_model(self, model_name, virtual=False, data_model=""):
        schema = {}
        if not virtual:
            for item in self.get_local_store("component"):
                if item.get("rec_name") == model_name:
                    schema = copy.deepcopy(item)
                    break
        await self.make_model(
            model_name,
            schema=schema,
            virtual=virtual,
            data_model=data_model or schema.get("data_model", ""),
        )

    async def make_model(
        self, model_name, schema: dict = None, virtual=False, data_model=""
    ):
        if schema is None:
            schema = {}
        if model_name in list(self.orm_static_models_map.keys()) or virtual:
            if not data_model and schema:
                data_model = schema.get("data_model", "")
            if (
                not data_model
                and not virtual
                and self.orm_static_models_map[model_name].get_data_model()
            ):
                data_model = self.orm_static_models_map[
                    model_name
                ].get_data_model()
            if data_model and virtual:
                data_model_o = self.env.models.get(data_model)
                if data_model_o and data_model_o.data_model:
                    data_model = data_model_o.data_model
            self.env.models[model_name] = self.cls_model(
                model_name,
                self,
                data_model=data_model,
                static=self.orm_static_models_map.get(model_name, None),
                virtual=virtual,
                schema=schema,
            )
            await self.env.models[model_name].init_model()

    async def update_model(self, schema, component):
        model_name = schema.get("rec_name")
        records = self.get_local_store("component")
        updated = False
        for idx, item in enumerate(records):
            if item.get("rec_name") == model_name:
                records[idx] = copy.deepcopy(schema)
                updated = True
                break
        if not updated:
            records.append(copy.deepcopy(schema))
        if model_name in self.orm_static_models_map:
            self.orm_static_models_map.pop(model_name)
        await self.init_model_and_write_code_from_schema(schema)
        await self.import_module_model(model_name)
        await self.make_model(
            model_name,
            schema=schema,
            virtual=False,
            data_model=schema.get("data_model", ""),
        )


class OzonModelRestBase(OzonModelBase):
    interface_type = "rest"

    def _is_local_only_model(self) -> bool:
        return self.orm.is_local_model(self.name)

    def _local_store(self) -> list[dict]:
        return self.orm.get_local_store(self.name)

    def _project_local_fields(self, record: dict, fields: dict) -> dict:
        if not fields:
            return copy.deepcopy(record)
        projected = {}
        include_keys = [key for key, enabled in fields.items() if enabled]
        if include_keys:
            for key in include_keys:
                if key in record:
                    projected[key] = record[key]
            return projected
        projected = copy.deepcopy(record)
        for key, enabled in fields.items():
            if enabled is False and key in projected:
                projected.pop(key)
        return projected

    def _sort_local_records(
        self, records: list[dict], sort: str
    ) -> list[dict]:
        sorted_records = list(records)
        sort_rules = list(self.eval_sort_str(sort).items())
        for key, direction in reversed(sort_rules):
            sorted_records.sort(
                key=lambda item: _read_record_value(item, key),
                reverse=direction < 0,
            )
        return sorted_records

    def _extract_response_data(self, response, default=None):
        if response is None:
            return default
        if isinstance(response, dict):
            if "data" in response:
                return response["data"]
            if "items" in response:
                return response["items"]
            if "result" in response:
                return response["result"]
            if "count" in response:
                return response["count"]
            if "value" in response:
                return response["value"]
        return response

    async def _post_remote(self, operation_name: str, payload: dict):
        return await self.orm.rest_client.post_operation(
            operation_name,
            payload=payload,
        )

    async def count_by_filter(self, domain: dict) -> int:
        self.init_status()
        if self._is_local_only_model():
            return len(
                [
                    item
                    for item in self._local_store()
                    if _match_local_domain(item, domain)
                ]
            )
        result = await self._post_remote(
            "count",
            {
                "model": self.name,
                "data_model": self.data_model,
                "domain": make_json_compatible(domain),
            },
        )
        data = self._extract_response_data(result, default=0)
        if isinstance(data, dict) and "count" in data:
            return int(data["count"])
        return int(data or 0)

    async def load_raw(self, domain: dict) -> Union[None, dict]:
        self.init_status()
        domain = traverse_and_convertd_datetime(domain)
        if self._is_local_only_model():
            for item in self._local_store():
                if _match_local_domain(item, domain):
                    return copy.deepcopy(item)
            self.error_status(_("Not found"), domain)
            return {}
        result = await self._post_remote(
            "load",
            {
                "model": self.name,
                "data_model": self.data_model,
                "domain": make_json_compatible(domain),
            },
        )
        data = self._extract_response_data(result, default={})
        if not data:
            self.error_status(_("Not found"), domain)
            return {}
        return data

    async def find_raw(
        self,
        domain: Optional[dict[str, Any]] = None,
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        pipeline_items: Optional[list[dict[str, Any]]] = None,
        obfuscate_fields: Optional[list[str]] = None,
        fields: Optional[dict] = None,
        batch_size: int = 0,
        need_cursor: bool = False,
    ):
        self.init_status()
        domain = traverse_and_convertd_datetime(domain or {})
        fields = fields or {}
        if self._is_local_only_model():
            records = [
                copy.deepcopy(item)
                for item in self._local_store()
                if _match_local_domain(item, domain)
            ]
            records = self._sort_local_records(records, sort)
            if skip > 0:
                records = records[skip:]
            if limit > 0:
                records = records[:limit]
            return [
                self._project_local_fields(item, fields) for item in records
            ]
        if pipeline_items or obfuscate_fields:
            return await self.aggregate_raw(
                domain=domain,
                sort=sort,
                limit=limit,
                skip=skip,
                pipeline_items=pipeline_items,
                obfuscate_fields=obfuscate_fields,
                fields=fields,
                batch_size=batch_size,
                need_cursor=need_cursor,
            )
        result = await self._post_remote(
            "find",
            {
                "model": self.name,
                "data_model": self.data_model,
                "domain": make_json_compatible(domain),
                "sort": sort,
                "limit": limit,
                "skip": skip,
                "fields": fields,
                "batch_size": batch_size,
            },
        )
        return self._extract_response_data(result, default=[])

    async def aggregate_raw(
        self,
        domain: Optional[dict[str, Any]] = None,
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        pipeline_items: Optional[list[dict[str, Any]]] = None,
        obfuscate_fields: Optional[list[str]] = None,
        fields: Optional[dict] = None,
        batch_size: int = 0,
        need_cursor: bool = False,
    ):
        self.init_status()
        result = await self._post_remote(
            "aggregate",
            {
                "model": self.name,
                "data_model": self.data_model,
                "domain": make_json_compatible(domain or {}),
                "sort": sort,
                "limit": limit,
                "skip": skip,
                "pipeline_items": make_json_compatible(pipeline_items or []),
                "obfuscate_fields": obfuscate_fields or [],
                "fields": fields or {},
                "batch_size": batch_size,
            },
        )
        return self._extract_response_data(result, default=[])

    async def insert(
        self, record: CoreModel, is_many=False
    ) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("User is Readonly")
            self.error_status(msg, data={})
            return None
        if self._is_local_only_model():
            record.create_datetime = record.utc_now()
            if self.user_session:
                record = self.set_user_data(record, self.user_session)
            if not is_many:
                record.list_order = await self.count()
            record.active = True
            self._local_store().append(record.get_dict_json())
            return await self.load(record.rec_name_domain())
        try:
            record_payload = await self._prepare_transport_record(
                record.get_dict()
            )
        except AttachmentError as e:
            self.error_status(str(e), record.get_dict_copy())
            return None
        result = await self._post_remote(
            "insert",
            {
                "model": self.name,
                "data_model": self.data_model,
                "record": make_json_compatible(record_payload),
                "is_many": is_many,
            },
        )
        data = self._extract_response_data(result, default={})
        if not data:
            self.error_status(
                _("Error save  %s ") % str(record.rec_name),
                record.get_dict_copy(),
            )
            return None
        await self.load_data(data)
        return self.modelr

    async def update(
        self,
        record: CoreModel,
        remove_mata=True,
        force_update_whole_record=False,
    ) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("User is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self._is_local_only_model():
            records = self._local_store()
            record_data = record.get_dict_json()
            record_data["update_datetime"] = record.utc_now().isoformat()
            if self.user_session:
                record_data["update_uid"] = self._user_uid(self.user_session)
            for idx, item in enumerate(records):
                if item.get("rec_name") == record.rec_name:
                    records[idx] = record_data
                    return await self.load(record.rec_name_domain())
            self.error_status(_("Not found"), record.rec_name_domain())
            return None
        try:
            record_payload = await self._prepare_transport_record(
                record.get_dict()
            )
        except AttachmentError as e:
            self.error_status(str(e), record.get_dict_copy())
            return None
        result = await self._post_remote(
            "update",
            {
                "model": self.name,
                "data_model": self.data_model,
                "record": make_json_compatible(record_payload),
                "remove_mata": remove_mata,
                "force_update_whole_record": force_update_whole_record,
            },
        )
        data = self._extract_response_data(result, default={})
        if not data:
            self.error_status(_("Not found"), record.rec_name_domain())
            return None
        await self.load_data(data)
        return self.modelr

    async def remove(self, record: CoreModel) -> bool:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("User is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return False
        if self._is_local_only_model():
            original_len = len(self._local_store())
            self.orm.local_store[self.name] = [
                item
                for item in self._local_store()
                if item.get("rec_name") != record.rec_name
            ]
            return len(self.orm.local_store[self.name]) != original_len
        result = await self._post_remote(
            "remove",
            {
                "model": self.name,
                "data_model": self.data_model,
                "record": record.get_dict_json(),
            },
        )
        data = self._extract_response_data(result, default=True)
        if isinstance(data, dict):
            return bool(data.get("deleted", data.get("ok", False)))
        return bool(data)

    async def remove_all(self, domain) -> int:
        self.init_status()
        domain = traverse_and_convertd_datetime(domain)
        if self._is_local_only_model():
            original = self._local_store()
            remain = [
                item
                for item in original
                if not _match_local_domain(item, domain)
            ]
            removed = len(original) - len(remain)
            self.orm.local_store[self.name] = remain
            return removed
        result = await self._post_remote(
            "remove_all",
            {
                "model": self.name,
                "data_model": self.data_model,
                "domain": make_json_compatible(domain),
            },
        )
        data = self._extract_response_data(result, default=0)
        if isinstance(data, dict) and "count" in data:
            return int(data["count"])
        return int(data or 0)

    async def distinct(self, field_name: str, query: dict) -> list[Any]:
        self.init_status()
        query = traverse_and_convertd_datetime(query)
        if self._is_local_only_model():
            values = []
            for item in self._local_store():
                if _match_local_domain(item, query):
                    values.append(_read_record_value(item, field_name))
            return list(dict.fromkeys(values))
        result = await self._post_remote(
            "distinct",
            {
                "model": self.name,
                "data_model": self.data_model,
                "field_name": field_name,
                "query": make_json_compatible(query),
            },
        )
        data = self._extract_response_data(result, default=[])
        return list(data or [])

    async def search_all_distinct(
        self,
        distinct: str = "",
        query: Optional[dict] = None,
        compute_label: str = "",
        sort: str = "",
        limit: int = 0,
        skip: int = 0,
        raw_result: bool = False,
    ) -> list[Any]:
        self.init_status()
        result = await self._post_remote(
            "search_all_distinct",
            {
                "model": self.name,
                "data_model": self.data_model,
                "distinct": distinct,
                "query": make_json_compatible(query or {}),
                "compute_label": compute_label,
                "sort": sort,
                "limit": limit,
                "skip": skip,
                "raw_result": raw_result,
            },
        )
        data = self._extract_response_data(result, default=[])
        if raw_result:
            return list(data or [])
        results = []
        for item in data or []:
            await self.load_data(item)
            results.append(self.modelr)
        return results

    async def stream_find(
        self,
        domain: Optional[dict[str, Any]] = None,
        sort: str = "",
        limit=0,
        skip=0,
        pipeline_items: Optional[list[dict[str, Any]]] = None,
        obfuscate_fields: Optional[list[str]] = None,
        fields: Optional[dict] = None,
        batch_size: int = 500,
    ) -> AsyncIterator[Any]:
        result = await self.find_raw(
            domain=domain,
            sort=sort,
            limit=limit,
            skip=skip,
            pipeline_items=pipeline_items,
            obfuscate_fields=obfuscate_fields,
            fields=fields,
            batch_size=batch_size,
            need_cursor=True,
        )
        for rec_data in result:
            modelr, _ = await self._load_data(
                self.model,
                rec_data,
                self.virtual,
                self.data_model,
                self.tz,
                self.virtual_fields_parser,
            )
            yield modelr


class OzonModel(OzonModelBase):
    def __init__(
        self,
        model_name,
        orm: OzonOrm,
        data_model="",
        virtual=False,
        static: CoreModel = None,
        schema={},
    ):
        self.orm: OzonOrm = orm
        self.env: OzonEnvBase = orm.env
        self.setting_app: Settings = orm.app_settings
        self.db: Mongo = orm.env.db
        self.mm_from_cache = False
        self.use_cache = False
        self.private_models = ["settings"]
        self.service: ModelService = None
        super(OzonModel, self).__init__(
            model_name=model_name,
            setting_app=self.setting_app,
            data_model=data_model,
            virtual=virtual,
            static=static,
            schema=schema,
        )

    @property
    def user_session(self):
        return self.orm.user_session

    def init_status(self):
        if self.user_session and self.user_session.is_public:
            if self.name.lower() in self.orm.private_models:
                raise OzonPermissionError(detail="Permission Denied")
        super().init_status()

    def chk_write_permission(self) -> bool:
        res = super().chk_write_permission()
        return res

    async def update(
        self,
        record: CoreModel,
        remove_mata=True,
        force_update_whole_record=False,
    ) -> Union[None, CoreModel]:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("User is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return None
        if self._transaction:
            original = await self.load_raw(record.rec_name_domain())
            self.env.local_transaction_add(
                self.data_model, "update", record.rec_name, original
            )
        return await super().update(
            record, remove_mata, force_update_whole_record
        )

    async def insert(
        self, record: CoreModel, is_many=False
    ) -> Union[None, CoreModel]:
        rec = await super().insert(record, is_many=is_many)
        if rec and self._transaction:
            self.env.local_transaction_add(
                self.data_model, "insert", rec.rec_name, {}
            )
        return rec

    async def remove(self, record: CoreModel) -> bool:
        self.init_status()
        if not self.chk_write_permission():
            msg = _("User is Readonly")
            self.error_status(msg, data=record.get_dict_json())
            return False
        if self._transaction:
            self.env.local_transaction_add(
                self.data_model, "delete", record.rec_name, record.get_dict()
            )
        return await super().remove(record)


class OzonModelRest(OzonModelRestBase):
    def __init__(
        self,
        model_name,
        orm: OzonOrmRest,
        data_model="",
        virtual=False,
        static: CoreModel = None,
        schema={},
    ):
        self.orm: OzonOrmRest = orm
        self.env: OzonEnvBase = orm.env
        self.setting_app: Settings = orm.app_settings
        self.db = None
        self.mm_from_cache = False
        self.use_cache = False
        self.private_models = ["settings"]
        self.service: ModelService = None
        super(OzonModelRest, self).__init__(
            model_name=model_name,
            setting_app=self.setting_app,
            data_model=data_model,
            virtual=virtual,
            static=static,
            schema=schema,
        )

    @property
    def user_session(self):
        return self.orm.user_session

    def init_status(self):
        if self.user_session and self.user_session.is_public:
            if self.name.lower() in self.orm.private_models:
                raise OzonPermissionError(detail="Permission Denied")
        super().init_status()

    def chk_write_permission(self) -> bool:
        return super().chk_write_permission()
