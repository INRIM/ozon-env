import datetime
import json
from decimal import Decimal
from typing import Any


class Coder:
    @classmethod
    def encode(cls, value: Any):
        raise NotImplementedError

    @classmethod
    def decode(cls, value: Any):
        raise NotImplementedError


class JsonCoder(Coder):
    @classmethod
    def encode(cls, value: Any):
        return json.dumps(value, default=cls._default_encoder).encode("utf-8")

    @classmethod
    def decode(cls, value: Any):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value, object_hook=cls._object_hook)

    @staticmethod
    def _default_encoder(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    @staticmethod
    def _object_hook(dct):
        for k, v in dct.items():
            if isinstance(v, str):
                try:
                    # Try to parse ISO datetime
                    datetime.datetime.fromisoformat(v)
                    dct[k] = v # Keep as string or convert? Usually safer to keep as string and convert at model level
                except (ValueError, TypeError):
                    pass
        return dct
