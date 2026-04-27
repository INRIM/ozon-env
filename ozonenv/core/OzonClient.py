import json
import logging
import os
from datetime import date, datetime

import aiofiles
import httpx

logger = logging.getLogger("asyncio")


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()


def make_json_compatible(data):
    return json.loads(json.dumps(data, default=json_serial))


class OzonClient:
    @classmethod
    def create(cls, apikey, is_api=False, url="http://client:8526"):
        self = OzonClient()
        self.default_url = url
        self.is_api = is_api
        self.api_key = apikey
        return self

    def get_headers(self):
        header = {
            "authtoken": f"{self.api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        if self.is_api:
            header.pop("authtoken")
            header["apitoken"] = self.api_key
        return header.copy()

    async def delete_attachment(self, field_key, model, rec_name, data):
        url = f"{self.default_url}/client/attachment/trash/{model}/{rec_name}"
        data_obj = {
            "field": field_key,
            "key": data.get("key"),
            "filename": data.get("filename"),
            "file_path": data.get("file_path"),
        }
        headers = self.get_headers()
        result = {"status": "ok"}
        async with httpx.AsyncClient(timeout=None) as client:
            res = await client.post(url, json=data_obj, headers=headers)
            if res:
                res = res.json()
                if isinstance(res, list) and len(res) > 0:
                    r = res[0]
                    if r.get("status") == "error":
                        result["status"] = "error"
                        return result
                return result
            else:
                return {"status": "error", "message": res}


class OzonDataApiClient:
    @classmethod
    def create(
        cls,
        base_url="",
        api_prefix="/base_usr/v2",
        token="",
        job_token="",
        oauth_url="",
        oauth_client_id="",
        oauth_client_secret="",
        token_audience="",
        timeout=90,
    ):
        self = OzonDataApiClient()
        self.base_url = str(base_url or "").rstrip("/")
        self.api_prefix = "/" + str(api_prefix or "base_usr/v2").strip("/")
        self.token = token or ""
        self.job_token = job_token or ""
        self.oauth_url = oauth_url or os.getenv("OZON_OAUTH_URL", "")
        self.oauth_client_id = (
            oauth_client_id
            or os.getenv("OZON_REST_CLIENT_ID", "")
            or os.getenv("OZON_M2M_CLIENT_ID", "")
            or os.getenv("OZON_CLIENT_ID", "")
        )
        self.oauth_client_secret = (
            oauth_client_secret
            or os.getenv("OZON_REST_CLIENT_SECRET", "")
            or os.getenv("OZON_M2M_CLIENT_SECRET", "")
            or os.getenv("OZON_CLIENT_SECRET", "")
        )
        self.token_audience = token_audience or os.getenv(
            "OZON_TOKEN_AUDIENCE", ""
        )
        self.timeout = timeout
        return self

    def set_token(self, token: str):
        self.token = token or ""

    def set_job_token(self, job_token: str):
        self.job_token = job_token or ""

    def has_oauth_config(self) -> bool:
        return any(
            [
                self.oauth_url,
                self.oauth_client_id,
                self.oauth_client_secret,
            ]
        )

    def can_generate_token(self) -> bool:
        return all(
            [
                self.oauth_url,
                self.oauth_client_id,
                self.oauth_client_secret,
            ]
        )

    async def _get_token(self) -> str:
        if not self.can_generate_token():
            raise ValueError(
                "OZON_OAUTH_URL, OZON_CLIENT_ID and "
                "OZON_CLIENT_SECRET are required to generate an OAuth token"
            )
        data = {
            "grant_type": "client_credentials",
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
        }
        if self.token_audience:
            data["audience"] = self.token_audience
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.oauth_url, data=data)
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError(
                "OAuth token response does not contain access_token"
            )
        return token

    async def ensure_token(self):
        if self.token:
            return
        if not self.has_oauth_config():
            return
        self.token = await self._get_token()

    def get_headers(self):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.job_token:
            headers["job_token"] = self.job_token
        return headers

    def operation_url(self, operation_name: str) -> str:
        return f"{self.base_url}{self.api_prefix}/{operation_name}"

    def resource_url(self, resource_path: str) -> str:
        return f"{self.base_url}{self.api_prefix}/{resource_path.lstrip('/')}"

    async def post_operation(self, operation_name: str, payload: dict = None):
        if payload is None:
            payload = {}
        await self.ensure_token()
        url = self.operation_url(operation_name)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=make_json_compatible(payload),
                headers=self.get_headers(),
            )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    async def get_resource(
        self,
        resource_path: str,
        params: dict = None,
    ):
        if params is None:
            params = {}
        await self.ensure_token()
        url = self.resource_url(resource_path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                url,
                params=make_json_compatible(params),
                headers=self.get_headers(),
            )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    async def delete_attachments(self, form_data, model, attachment_key):
        rec_name = form_data.get("rec_name")
        attachments = form_data.get(attachment_key)
        result = {"status": "ok", "message": "done"}
        for attachment in attachments:
            res = await self.delete_attachment(
                attachment_key, model, rec_name, attachment
            )
            if res.get("status") == "error":
                result["message"] = (
                    f"Error delete file"
                    f" {attachment['filename']} "
                    f"key {attachment['key']}"
                )
                result["status"] = "error"
                return result
        return result

    async def send_mail(self, model, rec_name, tmp_name):
        url = (
            f"{self.default_url}/client/send/"
            f"mail/{model}/{rec_name}/{tmp_name}"
        )
        data_obj = {}
        headers = self.get_headers()
        result = {"status": "ok"}
        async with httpx.AsyncClient(timeout=None) as client:
            res = await client.post(url, json=data_obj, headers=headers)
            if res:
                res = res.json()
                if isinstance(res, list) and len(res) > 0:
                    r = res[0]
                    if r.get("status") == "error":
                        result["status"] = "error"
                        return result
                return result
            else:
                return {"status": "error", "message": res}

    async def post_form_with_file(
        self, url, headers, form_data: dict = {}, files: list = []
    ):
        file_list = []
        headers.pop('content-type')
        for f_todo in files:
            f = await aiofiles.open(f_todo['file_path'], 'rb')
            data = await f.read()
            file_list.append((f_todo['file_key'], (f_todo['file_name'], data)))
            await f.close()
        client = httpx.AsyncClient(timeout=120)
        return await client.post(
            url,
            files=file_list,
            data={
                'formObj': json.dumps(
                    form_data, sort_keys=True, indent=1, default=json_serial
                )
            },
            headers=headers,
        )

    async def post_form_data(self, url, headers, form_data={}):
        client = httpx.AsyncClient(timeout=90)
        return await client.post(
            url,
            data={
                'formObj': json.dumps(
                    form_data, sort_keys=True, indent=1, default=json_serial
                )
            },
            headers=headers,
        )

    async def post_form(
        self, action_name, model, form_data: dict = None, files: list = None
    ):
        if files is None:
            files = []
        if form_data is None:
            form_data = {}
        url = f"{self.default_url}/" f"action/{action_name}"
        if form_data.get("rec_name"):
            url = f"{url}/{form_data.get('rec_name')}"
        headers = self.get_headers()
        form_data['data_model'] = model
        if files:
            res = await self.post_form_with_file(
                url, headers, form_data, files
            )
        else:
            res = await self.post_form_data(url, headers, form_data)
        result = {"status": "ok"}
        if res:
            res = res.json()
            if isinstance(res, list) and len(res) > 0:
                r = res[0]
                if r.get("status") == "error":
                    result["status"] = "error"
                    return result
            elif isinstance(res, dict):
                if "data" in res:
                    result["data"] = res["data"]
                    return result

            return result
        else:
            return {"status": "error", "message": res}

    async def copy_attachments(
        self,
        model: str,
        rec_name: str,
        field: str,
        dest: str,
    ) -> dict:
        url = (
            f"{self.default_url}/client/attachment/"
            f"copy/{model}/{rec_name}/{field}/{dest}"
        )
        headers = self.get_headers()
        async with httpx.AsyncClient(timeout=None) as client:
            res = await client.post(url, json={}, headers=headers)
            if res:
                return res
            else:
                return {"status": "error", "message": res}

    async def unlink_attachment(self, field_key, model, rec_name, data):
        url = (
            f"{self.default_url}/client/"
            f"attachment/unlink/{model}/{rec_name}"
        )
        data_obj = {
            "field": field_key,
            "key": data.get("key"),
            "filename": data.get("filename"),
            "file_path": data.get("file_path"),
        }
        headers = self.get_headers()
        result = {"status": "ok"}
        async with httpx.AsyncClient(timeout=None) as client:
            res = await client.post(url, json=data_obj, headers=headers)
            if res:
                res = res.json()
                if isinstance(res, list) and len(res) > 0:
                    r = res[0]
                    if r.get("status") == "error":
                        result["status"] = "error"
                        return result
                return result
            else:
                return {"status": "error", "message": res}


class LabelPrinter:
    @classmethod
    def create(cls, apikey="", is_api=False, url=""):
        self = LabelPrinter()
        self.default_url = url
        self.is_api = is_api
        self.api_key = apikey
        return self

    def get_headers(self):
        header = {
            "authtoken": f"{self.api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        if self.is_api:
            header.pop("authtoken")
            header["apitoken"] = self.api_key
        return header.copy()

    async def status(
        self,
    ):
        url = f"{self.default_url}/status"
        logger.info(url)
        headers = self.get_headers()

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.get(url, headers=headers)
                return resp.json()
        except Exception as e:
            logger.error(e, exc_info=True)
            return {"status": "error", "message": str(e)}

    async def print_label(self, payload):
        url = f"{self.default_url}/print_label"
        logger.info(url)
        headers = self.get_headers()

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(url, json=payload, headers=headers)
                return resp.json()
        except Exception as e:
            logger.error(e, exc_info=True)
            return {"status": "error", "message": str(e)}
