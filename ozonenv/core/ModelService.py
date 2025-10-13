import json
import locale
from dataclasses import field
from datetime import datetime
from typing import Union, Any, Optional, List

from pydantic import AwareDatetime

from ozonenv.core.BaseModels import CoreModel, defaultdt
from ozonenv.core.DateEngine import DateEngine


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

    def eval_datetime(self, value: AwareDatetime, name: str = None):
        dttype = self.tranform.get(name, {}).get("type", "datetime")
        if dttype == "datetime":
            return self.dte.to_ui(
                value, self.tranform.get(name, {}).get("type", "datetime")
            )
        else:
            return self.dte.to_ui(
                value, self.tranform.get(name, {}).get("type", "date")
            )

    def eval_float(self, value, name):
        return self.readable_float(
            value,
            self.tranform.get(name, {}).get("dp", 2),
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
            return res

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
        data_value = {}
        if pdata_value is not None:
            data_value = pdata_value.copy()
        for name, field in self.model.model_fields.items():
            if name not in dati:
                continue
            if field.annotation in (
                datetime,
                AwareDatetime,
                Optional[AwareDatetime],
            ):
                res = self.eval_datetime(dati.get(name, defaultdt), name)

                data_value = self.check_update_data_value(
                    name, data_value, res
                )

            elif field.annotation in (float, Optional[float]):
                res = self.eval_float(dati.get(name, 0.0), name)
                data_value = self.check_update_data_value(
                    name, data_value, res
                )

            elif name in self.model.select_fields():
                res = await self.eval_select(name, dati[name])
                data_value = self.check_update_data_value(
                    name, data_value, res
                )

        dati["data_value"] = data_value
        return dati.copy()
