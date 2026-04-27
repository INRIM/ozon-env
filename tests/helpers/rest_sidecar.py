import asyncio
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ozonenv.OzonEnv import OzonEnv
from ozonenv.core.OzonClient import make_json_compatible
from tests.test_common import init_main_collections


def strip_mongo_ids(value):
    if isinstance(value, list):
        return [strip_mongo_ids(item) for item in value]
    if isinstance(value, dict):
        data = {
            key: strip_mongo_ids(item)
            for key, item in value.items()
            if key != "_id"
        }
        return data
    return value


def build_test_db_cfg(
    models_folder: Path, app_code: str | None = None
) -> dict:
    return {
        "app_code": app_code or os.getenv("APP_CODE"),
        "mongo_user": os.getenv("MONGO_USER"),
        "mongo_pass": os.getenv("MONGO_PASS"),
        "mongo_url": os.getenv("MONGO_URL"),
        "mongo_db": os.getenv("MONGO_DB"),
        "mongo_replica": os.getenv("MONGO_REPLICA"),
        "models_folder": str(models_folder),
        "keycloak_jwks_url": os.getenv("OZON_KEYCLOAK_JWKS_URL"),
        "keycloak_issuer": os.getenv("OZON_KEYCLOAK_ISSUER"),
        "token_audience": os.getenv("OZON_TOKEN_AUDIENCE"),
        "oauth_url": os.getenv("OZON_OAUTH_URL"),
        "client_id": os.getenv("OZON_CLIENT_ID"),
        "client_secret": os.getenv("OZON_CLIENT_SECRET"),
        "rest_client_id": os.getenv("OZON_M2M_CLIENT_ID"),
        "rest_client_secret": os.getenv("OZON_M2M_CLIENT_SECRET"),
    }


class RealOzonEnvApiServer:
    def __init__(self, cfg: dict = {}, api_prefix: str = "/v2"):
        self.cfg = cfg
        self.api_prefix = api_prefix
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self):
        handler_class = self._make_handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
        )
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=5)

    def _make_handler(self):
        cfg = copy.deepcopy(self.cfg)
        api_prefix = self.api_prefix

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                self._run_request(self._handle_get())

            def do_POST(self):
                self._run_request(self._handle_post())

            def _run_request(self, coro):
                try:
                    status, payload = asyncio.run(coro)
                except Exception as exc:
                    status = 500
                    payload = {"error": str(exc)}
                self._send_json(status, payload)

            def _send_json(self, status: int, payload):
                body = json.dumps(
                    make_json_compatible(strip_mongo_ids(payload)),
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _operation_name(self) -> str:
                path = self.path.split("?", 1)[0]
                if not path.startswith(api_prefix):
                    return ""
                return path[len(api_prefix) :].strip("/")

            async def _create_env(self, validate_auth: bool = True):
                env = OzonEnv()
                await env.init_env()
                try:
                    await init_main_collections(env.db)
                    if validate_auth:
                        await env.init_api_job_context(
                            self.headers.get("Authorization", ""),
                            self.headers.get("job_token", ""),
                        )
                    return env
                except Exception:
                    await env.close_env()
                    raise

            async def _handle_get(self):
                operation = self._operation_name()
                if not operation:
                    return 404, {"error": "Not found"}
                env = await self._create_env(validate_auth=True)
                try:
                    if operation == "collections_names":
                        return 200, {
                            "data": await env.orm.get_collections_names()
                        }
                    if operation.startswith("init_settings/"):
                        app_code = operation.split("/", 1)[1]
                        settings = await env.orm.init_settings(app_code)
                        return 200, {"data": settings.get_dict_json()}
                    return 404, {"error": "Not found"}
                finally:
                    await env.close_env()

            async def _handle_post(self):
                operation = self._operation_name()
                if not operation:
                    return 404, {"error": "Not found"}
                length = int(self.headers.get("content-length", "0") or 0)
                raw_body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                env = await self._create_env(validate_auth=True)
                try:
                    model_name = payload.get("model", "")
                    model = env.get(model_name)
                    if not model:
                        return 404, {"error": f"Model {model_name} not found"}
                    if operation == "find":
                        records = await model.find_raw(
                            payload.get("domain") or {},
                            sort=payload.get("sort", ""),
                            limit=int(payload.get("limit") or 0),
                            skip=int(payload.get("skip") or 0),
                            fields=payload.get("fields") or {},
                            batch_size=int(payload.get("batch_size") or 0),
                        )
                        return 200, {"data": records}
                    if operation == "load":
                        record = await model.load_raw(
                            payload.get("domain") or {}
                        )
                        return 200, {"data": record}
                    if operation == "count":
                        count = await model.count_by_filter(
                            payload.get("domain") or {}
                        )
                        return 200, {"data": {"count": count}}
                    if operation == "insert":
                        record = await model.new(payload.get("record") or {})
                        saved = await model.insert(
                            record,
                            is_many=bool(payload.get("is_many", False)),
                        )
                        return 200, {
                            "data": saved.get_dict_json() if saved else {}
                        }
                    return 404, {"error": "Not found"}
                finally:
                    await env.close_env()

        return Handler
