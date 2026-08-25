import copy

import pytest
from pydantic import ValidationError

from ozonenv.core.BaseModels import BasicModel, CoreNestedModel, Settings
from ozonenv.core.OzonOrm import OzonModel


class NestedInput(CoreNestedModel):
    code: str


class NormalizedInput(BasicModel):
    text: str
    optional_text: str | None = None
    count: int
    ratio: float
    enabled: bool
    nested: NestedInput
    rows: list[NestedInput]


class DummyEnv:
    db = None

    @staticmethod
    def is_data_value_runtime_enabled(_model_name: str) -> bool:
        return False


class DummyOrm:
    def __init__(self):
        self.env = DummyEnv()
        self.app_settings = Settings(
            rec_name="test",
            tz="Europe/Rome",
        )
        self.user_session = None
        self.private_models = []


def test_scalar_strings_are_normalized_from_numeric_values():
    payload = {
        "text": 465,
        "optional_text": 12.5,
        "count": 7,
        "ratio": 1.5,
        "enabled": True,
        "nested": {"code": 99},
        "rows": [{"code": False}, {"code": "ready"}],
    }

    normalized = NormalizedInput.normalize_model_fields(payload)
    model = NormalizedInput(**normalized)

    assert type(normalized["count"]) is int
    assert type(normalized["ratio"]) is float
    assert type(normalized["enabled"]) is bool
    assert model.text == "465"
    assert model.optional_text == "12.5"
    assert model.count == 7
    assert model.ratio == 1.5
    assert model.enabled is True
    assert model.nested.code == "99"
    assert [row.code for row in model.rows] == ["False", "ready"]


def test_valid_strings_and_none_are_preserved_without_mutating_input():
    payload = {
        "text": "already-valid",
        "optional_text": None,
        "count": 7,
        "ratio": 1.5,
        "enabled": False,
        "nested": {"code": 10},
        "rows": [],
    }
    original = copy.deepcopy(payload)

    normalized = NormalizedInput.normalize_model_fields(payload)

    assert normalized["text"] == "already-valid"
    assert normalized["optional_text"] is None
    assert payload == original
    assert normalized is not payload
    assert normalized["nested"] is not payload["nested"]


@pytest.mark.parametrize("invalid_value", [{"value": 1}, [1, 2]])
def test_complex_values_for_string_fields_still_fail_validation(invalid_value):
    payload = {
        "text": invalid_value,
        "optional_text": None,
        "count": 7,
        "ratio": 1.5,
        "enabled": True,
        "nested": {"code": "nested"},
        "rows": [],
    }

    normalized = NormalizedInput.normalize_model_fields(payload)

    assert normalized["text"] == invalid_value
    with pytest.raises(ValidationError):
        NormalizedInput(**normalized)


@pytest.mark.asyncio
async def test_ozon_model_load_data_normalizes_before_pydantic_validation():
    model = OzonModel(
        "normalized_input",
        DummyOrm(),
        static=NormalizedInput,
    )
    await model.init_model()
    payload = {
        "rec_name": "record.one",
        "text": 465,
        "optional_text": 12,
        "count": 7,
        "ratio": 1.5,
        "enabled": True,
        "nested": {"code": 99},
        "rows": [{"code": 100}],
    }
    original = copy.deepcopy(payload)

    await model.load_data(payload)

    assert model.modelr.text == "465"
    assert model.modelr.optional_text == "12"
    assert model.modelr.nested.code == "99"
    assert model.modelr.rows[0].code == "100"
    assert payload == original


@pytest.mark.asyncio
async def test_dynamic_formio_select_accepts_numeric_frontend_value():
    schema = {
        "components": [
            {
                "type": "select",
                "key": "delivery_channel",
                "label": "Delivery channel",
                "input": True,
                "multiple": False,
                "dataSrc": "values",
                "data": {
                    "values": [
                        {"label": "Primary", "value": 465},
                    ]
                },
            }
        ]
    }
    model = OzonModel("dynamic_select", DummyOrm(), schema=schema)
    await model.init_model()

    assert model.model.model_fields["delivery_channel"].annotation is str

    await model.load_data(
        {"rec_name": "record.select", "delivery_channel": 465}
    )

    assert model.modelr.delivery_channel == "465"


class EmptyContainerInput(BasicModel):
    so: str = ""
    tags: list = []
    payload: dict = {}
    mixed: str | dict = ""


def test_empty_containers_on_string_fields_become_empty_string():
    """Il bottone "Fatto" di supervised_todo gira con
    showValidations=false: la submission formio arriva non validata e i
    componenti a valore-oggetto vuoti valgono {} / [], non "". Su un campo
    solo-str e' il vuoto."""
    normalized = EmptyContainerInput.normalize_model_fields(
        {"so": {}, "tags": [], "payload": {}}
    )

    assert normalized["so"] == ""
    # I campi che il model dichiara container restano container.
    assert normalized["tags"] == []
    assert normalized["payload"] == {}
    assert EmptyContainerInput(**normalized).so == ""


def test_non_empty_container_on_string_field_still_fails_loudly():
    """Un dict con dentro qualcosa e' un dato da mappare (valueProperty /
    select_fields), non un vuoto: non va inghiottito."""
    normalized = EmptyContainerInput.normalize_model_fields(
        {"so": {"value": "x", "label": "X"}}
    )

    assert normalized["so"] == {"value": "x", "label": "X"}
    with pytest.raises(ValidationError):
        EmptyContainerInput(**normalized)


def test_union_with_container_keeps_the_empty_container():
    normalized = EmptyContainerInput.normalize_model_fields({"mixed": {}})

    assert normalized["mixed"] == {}
