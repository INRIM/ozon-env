from typing import Any, List, Union

from pydantic import AwareDatetime

from ozonenv.core.BaseModels import CoreModel
from ozonenv.core.DataValueService import DataValueService
from ozonenv.core.DateEngine import DateEngine

# from ozonenv.core.OzonOrm import OzonOrm


class ModelService:
    def __init__(self, model: CoreModel, orm, tz):
        self.model: CoreModel = model
        self.orm = orm
        self.tranform = self.model.tranform_data_value()
        self.dte = DateEngine(TZ=tz)
        self.data = {}
        self.data_value_service = DataValueService(
            model=self.model,
            orm=self.orm,
            dte=self.dte,
        )

    def readable_float(self, val, dp=2, g=True):
        return self.data_value_service.readable_float(val, dp=dp, g=g)

    def eval_datetime(
        self,
        value: AwareDatetime,
        name: str = None,
        transform_config: dict = None,
    ):
        return self.data_value_service.eval_datetime(
            value=value,
            name=name,
            transform_config=transform_config,
        )

    def eval_float(self, value, name, transform_config: dict = None):
        return self.data_value_service.eval_float(
            value=value,
            name=name,
            transform_config=transform_config,
        )

    async def select_values(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, list]:
        return await self.data_value_service.select_values(
            select=select,
            options=options,
            value=value,
        )

    async def select_url_values(
        self, select: dict, value: Union[str, list]
    ) -> list[dict[str, Any]]:
        return await self.data_value_service.select_url_values(
            select=select, value=value
        )

    async def select_url(
        self, select: dict, options: list, value: Union[str, list]
    ) -> List[dict[str, str]]:
        return await self.data_value_service.select_url(
            select=select,
            options=options,
            value=value,
        )

    async def select_custom(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        return await self.data_value_service.select_custom(
            select=select,
            options=options,
            value=value,
        )

    async def select_resource(
        self, select: dict, options: list, value: Union[str, list]
    ) -> Union[str, List[dict[str, str]]]:
        return await self.data_value_service.select_resource(
            select=select,
            options=options,
            value=value,
        )

    async def eval_select(self, name, value):
        return await self.data_value_service.eval_select(
            name=name,
            value=value,
        )

    def check_update_data_value(self, name, data_value, value):
        return self.data_value_service.check_update_data_value(
            name=name,
            data_value=data_value,
            value=value,
        )

    async def compute_data_value(self, dati: dict, pdata_value: dict = None):
        return await self.data_value_service.compute_data_value(
            dati=dati,
            pdata_value=pdata_value,
        )
