import json
import locale
from dataclasses import field
from datetime import datetime
from typing import Union, Any, Optional, List, get_origin, get_args

from pydantic import AwareDatetime, BaseModel

from ozonenv.core.BaseModels import CoreModel, defaultdt, MainModel
from ozonenv.core.DateEngine import DateEngine
from ozonenv.core.utils import unwrap_optional

# from ozonenv.core.OzonOrm import OzonOrm


class ModelService:
    def __init__(self, model: CoreModel, orm, tz):
        self.model: CoreModel = model
        self.orm = orm
        self.tranform = self.model.tranform_data_value()
        self.dte = DateEngine(TZ=tz)
        self.data = {}

    def readable_float(self, val, dp=2, g=True):
        if isinstance(val, str):
            val = float(val)
        return locale.format_string(f"%.{dp}f", val, g)

    def eval_datetime(
        self,
        value: AwareDatetime,
        name: str = None,
        transform_config: dict = None,
    ):
        transform_dv_config = transform_config or self.tranform
        dttype = transform_dv_config.get(name, {}).get("type", "datetime")
        if dttype == "datetime":
            return self.dte.to_ui(
                value,
                transform_dv_config.get(name, {}).get("type", "datetime"),
            )
        else:
            return self.dte.to_ui(
                value, transform_dv_config.get(name, {}).get("type", "date")
            )

    def eval_float(self, value, name, transform_config: dict = None):
        transform_dv_config = transform_config or self.tranform
        return self.readable_float(
            value,
            transform_dv_config.get(name, {}).get("dp", 2),
        )

    async def select_values(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, list]:
        if not select['multi']:
            vals = [opt['label'] for opt in options if opt['value'] == value]
            val = vals and vals[0] or ''
            return val
        else:
            res = []
            for i in value:
                vals = [opt['label'] for opt in options if opt['value'] == i]
                vals and res.append(vals[0])
            return ", ".join(res)

    async def select_url_values(
        self, select: dict, value: Union[str, list]
    ) -> list[dict[str, Any]]:
        props = select.get('properties', {})
        if select['url'] == "/models/distinct":
            fdomain = json.loads(props.get('domain', '{}'))
            feld = props.get('id', 'rec_name')
            compute_label = props.get('compute_label', '')
            modelname = props.get('model')
            sort = props.get('sort', f'{field}:asc')
            limit = int(props.get('limit', '0'))
            skip = int(props.get('skip', '0'))
            model = self.orm.env.models.get(modelname)
            query = {
                "$and": [
                    model.default_domain.copy(),
                    fdomain,
                    {feld: value},
                ]
            }

            res = await model.search_all_distinct(
                distinct=feld,
                query=query,
                compute_label=compute_label,
                sort=sort,
                limit=limit,
                skip=skip,
                raw_result=True,
            )
            if not res:
                return []
            options = []
            for item in res:
                label_value = props["label"]
                label_values = label_value.split(",")
                if len(label_values) > 1:
                    lst_vals = [item[lv] for lv in label_values]
                    label = " ".join(lst_vals)
                else:
                    label = item[label_value]
                iid = item[props["id"]]
                options.append({"label": label, "value": iid})
            return options
        else:
            return []

    async def select_url(
        self, select: dict, options: list, value: Union[str, list]
    ) -> List[dict[str, str]]:
        options = await self.select_url_values(select, value)
        return await self.select_values(select, options, value)

    async def select_custom(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        # options = await self.model.select_custom(select, value)
        # return await self.select_values(select, options, value)
        if not select['multi']:
            return ""
        else:
            return []

    async def select_resource(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        # options = await self.model.select_custom(select, value)
        # return await self.select_values(select, options, value)
        if not select['multi']:
            return ""
        else:
            return []

    async def eval_select(self, name, value):
        select = self.model.select_fields()[name].copy()
        options = self.model.select_options()[name]
        return await getattr(self, f"select_{select['src']}")(
            select, options, value
        )

    def check_update_data_value(self, name, data_value, value):
        if name not in data_value:
            data_value[name] = value
        elif not data_value.get(name) and value:
            data_value[name] = value
        elif data_value.get(name) and value:
            data_value[name] = value
        return data_value.copy()

    async def compute_data_value(self, dati: dict, pdata_value: dict = None):
        data_value = pdata_value.copy() if pdata_value else {}

        async def _compute_model_fields(
            model: MainModel,
            input_data: dict,
            local_data_value: dict,
            nested_field: str = None,
        ) -> dict:

            for name, field in model.model_fields.items():
                if name not in input_data:
                    continue

                raw_value = input_data[name]
                actual_type = unwrap_optional(field.annotation)

                # --- Single nested Pydantic model ---
                if isinstance(raw_value, dict) and hasattr(
                    actual_type, "model_fields"
                ):
                    actual_dv = raw_value.get("data_value", {})
                    nested_result = await _compute_model_fields(
                        actual_type, raw_value, actual_dv, name
                    )
                    input_data[name]["data_value"] = nested_result
                    continue

                # --- List/Tuple of nested Pydantic models ---
                origin = get_origin(actual_type)
                args = get_args(actual_type)
                if origin in (list, tuple) and args:
                    elem_type = unwrap_optional(args[0])
                    if isinstance(raw_value, list) and hasattr(
                        elem_type, "model_fields"
                    ):
                        for idx, item in enumerate(raw_value):
                            if isinstance(item, dict):
                                actual_dv = item.get("data_value", {})
                                el_dv = await _compute_model_fields(
                                    elem_type, item, actual_dv, name
                                )
                                input_data[name][idx]["data_value"] = el_dv
                        continue

                # --- EXISTING LOGIC for primitive fields ---
                if actual_type in (datetime, AwareDatetime):
                    nested_trnsfm_dv = (
                        self.model.nested_transform_data_value().get(
                            nested_field, {}
                        )
                        if nested_field
                        else None
                    )
                    res = self.eval_datetime(
                        input_data.get(name, defaultdt), name, nested_trnsfm_dv
                    )
                    local_data_value = self.check_update_data_value(
                        name, local_data_value, res
                    )

                elif actual_type is float:
                    nested_trnsfm_dv = (
                        self.model.nested_transform_data_value().get(
                            nested_field, {}
                        )
                        if nested_field
                        else None
                    )
                    res = self.eval_float(
                        input_data.get(name, 0.0), name, nested_trnsfm_dv
                    )
                    local_data_value = self.check_update_data_value(
                        name, local_data_value, res
                    )
                elif (
                    hasattr(self.model, "select_fields")
                    and name in self.model.select_fields()
                ):
                    res = await self.eval_select(name, input_data[name])
                    local_data_value = self.check_update_data_value(
                        name, local_data_value, res
                    )

            return local_data_value

        data_value.update(
            await _compute_model_fields(self.model, dati, data_value)
        )
        dati["data_value"] = data_value
        return dati.copy()
