import logging
import os

from ozonenv.core.BaseModels import BasicReturn
from ozonenv.core.OzonClient import OzonClient
from ozonenv.core.OzonOrm import OzonEnvBase, OzonModel

logger = logging.getLogger(__file__)


class OzonEnv(OzonEnvBase):
    def __init__(self, cfg={}, upload_folder="", cls_model=OzonModel):
        super().__init__(
            cfg=cfg, upload_folder=upload_folder, cls_model=cls_model)


class OzonWorkerEnv(OzonEnv):
    def __init__(self, cfg={}, upload_folder="", cls_model=OzonModel):
        super(OzonWorkerEnv, self).__init__(
            cfg=cfg, upload_folder=upload_folder, cls_model=cls_model)
        self.doc_type = ""
        self.action_next_page = ""
        self.topic_name = ""
        self.params = {}

    @classmethod
    def next_client_url(
            cls, params: dict,
            type_resp="success",
            rec_ref="",
            default_url="self") -> str:
        action_next_page = default_url
        if params.get("action_next_page"):
            next_page_cg = params.get("action_next_page", {}).get(
                type_resp, {})
            if next_page_cg:
                if next_page_cg.get("list"):
                    action_next_page = next_page_cg.get('list')
                elif next_page_cg.get("form"):
                    action_next_page = f"{next_page_cg.get('form')}/{rec_ref}"
        return action_next_page

    @classmethod
    def worker_exception_response(
            cls, topic, doc_type, model, err,
            err_details="",
            redirect_url="self"):
        result = {
            "done": True,
            "error": True,
            "msg": err,
            "document_type": doc_type,
            "model": model,
            "next_action": "redirect",
            "next_page": redirect_url
        }

        return cls.fail_response(
            err,
            err_details=err_details,
            data={
                topic: result
            })

    @classmethod
    def worker_success_default_response(
            cls, topic, doc_type, model, msg="", redirect_url="self"):
        result = {
            "done": True,
            "error": False,
            "msg": msg,
            "document_type": doc_type,
            "model": model,
            "next_action": "redirect",
            "next_page": redirect_url
        }

        return cls.success_response(msg=msg, data={
            topic: result
        })

    def exception_response(
            self, err, err_details="", redirect_url="self",
            rec_ref=""):
        return self.worker_exception_response(
            self.topic_name, self.doc_type, self.model,
            err, err_details=err_details,
            redirect_url=self.next_client_url(
                params=self.params, type_resp="error",
                default_url=redirect_url,
                rec_ref=rec_ref)
        )

    def default_response(self, msg="", redirect_url="self", rec_ref=""):
        return self.worker_success_default_response(
            self.topic_name, self.doc_type, self.model,
            msg=msg,
            redirect_url=self.next_client_url(
                params=self.params, default_url=redirect_url, rec_ref=rec_ref)
        )

    async def make_app_session(
            self,
            params: dict,
            use_cache=False,
            cache_idx="ozon_env",
            redis_url="redis://redis_cache",
            db=None,
            local_model={}
    ) -> BasicReturn:
        self.topic_name = params.get('topic_name', "")
        res = await super(OzonWorkerEnv, self).make_app_session(
            params, use_cache=use_cache, cache_idx=cache_idx,
            redis_url=redis_url, db=db, local_model=local_model)
        if res.fail:
            return self.exception_response(res.msg)
        return res

    async def session_app(self) -> BasicReturn:
        res = await super(OzonWorkerEnv, self).session_app()
        if res.fail:
            return self.exception_response(res.msg)
        self.doc_type = self.params.get('document_type', "")
        self.topic_name = self.params.get('topic_name', "")
        self.model = self.params.get('model', "")
        self.ozon_client = OzonClient.create(
            self.session_token, is_api=self.session_is_api,
            url=os.getenv("OZON_CLIENT", "http://client:8526")
        )
        return self.default_response(msg="Done")
