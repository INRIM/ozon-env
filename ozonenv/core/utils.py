import base64
import tempfile
import re
from typing import get_origin, Union, get_args

import aiofiles
import aiofiles.os
import json
import httpx
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import warnings
from functools import wraps


def deprecated(reason: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


async def read_json_file(file_path):
    async with aiofiles.open(file_path, mode="r") as f:
        data = await f.read()
    return json.loads(data)


def base64_encode_url(url):
    content = httpx.get(url).content
    tf = tempfile.TemporaryFile()
    tf.write(content)
    tf.seek(0)
    b64encode = base64.b64encode(tf.read())
    tf.close()
    # prefix and decode bytes to str
    b64encode = "%s,%s" % ("data:image/png;base64", b64encode.decode())
    return b64encode


def decode_resource_template(tmp):
    res = re.sub(r"<.*?>", " ", tmp)
    strcleaned = re.sub(r"\{{ |\ }}", "", res)
    list_kyes = strcleaned.strip().split(".")
    return list_kyes[1:]


def fetch_dict_get_value(dict_src, list_keys):
    if len(list_keys) == 0:
        return
    node = list_keys[0]
    list_keys.remove(node)
    nextdict = dict_src.get(node)
    if len(list_keys) >= 1:
        return fetch_dict_get_value(nextdict, list_keys)
    else:
        return dict_src.get(node)


def is_json(str_test):
    try:
        str_test = json.loads(str_test)
    except ValueError:
        str_test = str_test.replace("'", '"')
        try:
            str_test = json.loads(str_test)
        except ValueError:
            return False
    return str_test


def log_truncate(value, maxlen=20):
    text = str(value)
    return text if len(text) <= maxlen else text[: maxlen - 3] + "..."


ISO_DATE_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}([T\s]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}(:?\d{2})?)?)?$'
)


def safe_parse_datetime(value: str):
    """Tenta di convertire una stringa ISO8601 in datetime, in modo tollerante."""
    if not isinstance(value, str):
        return value
    if not ISO_DATE_RE.match(value):
        return value
    try:
        # supporta sia con Z che con offset
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("UTC"))
    except ValueError:
        # fallback: nessun parsing riuscito → restituisci originale
        return value


def traverse_and_convertd_datetime(obj):
    """
    Attraversa ricorsivamente dict, list, tuple, set, convertendo stringhe data/ora in datetime.
    """
    if isinstance(obj, dict):
        return {k: traverse_and_convertd_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [traverse_and_convertd_datetime(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(traverse_and_convertd_datetime(i) for i in obj)
    elif isinstance(obj, set):
        return {traverse_and_convertd_datetime(i) for i in obj}
    elif isinstance(obj, str):
        return safe_parse_datetime(obj)
    else:
        return obj


def unwrap_optional(annotation):
    """Return the actual type inside Optional / Union[..., None]"""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return args[0]
    return annotation
