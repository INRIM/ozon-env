from pydantic._internal._model_construction import ModelMetaclass

from ozonenv.core.ModelMaker import ModelMaker, MainModel
from test_common import *

# from ozonenv.core.i18n import i18nlocaledir
pytestmark = pytest.mark.asyncio


@pytestmark
async def test_make_form_data():
    data_json = await get_file_data()
    test_1 = ModelMaker("test_base", tz="Europe/Rome")
    test_1.from_data_dict(data_json)
    test_1.new()
    assert test_1.model_name == "test_base"
    assert isinstance(test_1.instance, BasicModel) is True
    assert test_1.instance.annoRif == 2022
    assert test_1.instance.dg11XContr.flRate is True
    assert test_1.instance.dg11XContr.get('flRate') is True
    assert len(test_1.instance.dg15XVoceCalcolata) == 4
    assert test_1.instance.dg15XVoceCalcolata[1].get('importo') == 289.23


@pytestmark
async def test_make_form_schema():
    schema = await get_formio_schema()
    formio_data_json = await get_formio_data()
    test_2 = ModelMaker("component", tz="Europe/Rome")
    test_2.from_formio(schema)
    assert test_2.model_name == "component"
    assert isinstance(test_2.model, ModelMetaclass) is True
    assert test_2.unique_fields == ["rec_name", "firstName"]
    assert test_2.required_fields == ["rec_name", "firstName"]
    assert test_2.components_logic == []
    assert test_2.datetime_fields == {
        'update_datetime': {'transform': {'type': 'datetime'}},
        'create_datetime': {'transform': {'type': 'datetime'}},
        'birthdate': {
            'ctype': 'datetime',
            'disabled': False,
            'readonly': False,
            'hidden': False,
            'required': False,
            'unique': False,
            'component': 'Component',
            'calculateServer': '',
            'action_type': False,
            'no_clone': False,
            'transform': {'type': 'date'},
            'datetime': False,
            'min': None,
            'max': None,
            'time': False,
            'date': True,
        },
        'appointmentDateTime': {
            'ctype': 'datetime',
            'disabled': False,
            'readonly': False,
            'hidden': False,
            'required': False,
            'unique': False,
            'component': 'Component',
            'calculateServer': '',
            'action_type': False,
            'no_clone': False,
            'transform': {'type': 'datetime'},
            'datetime': True,
            'min': None,
            'max': None,
            'time': True,
            'date': True,
        },
        'appointmentDateTime1': {
            'ctype': 'datetime',
            'disabled': False,
            'readonly': False,
            'hidden': False,
            'required': False,
            'unique': False,
            'component': 'Component',
            'calculateServer': '',
            'action_type': False,
            'no_clone': False,
            'transform': {'type': 'datetime'},
            'datetime': True,
            'min': None,
            'max': None,
            'time': True,
            'date': True,
        },
    }
    assert test_2.tranform_data_value == {
        'birthdate': {'type': 'date'},
        'appointmentDateTime': {'type': 'datetime'},
        'appointmentDateTime1': {'type': 'datetime'},
    }
    assert test_2.nested_datetime_fields == {
        'dataGrid': {
            'update_datetime': {'transform': {'type': 'datetime'}},
            'create_datetime': {'transform': {'type': 'datetime'}},
            'birthdateDg': {
                'ctype': 'datetime',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': '',
                'action_type': False,
                'no_clone': False,
                'transform': {'type': 'date'},
                'datetime': False,
                'min': None,
                'max': None,
                'time': False,
                'date': True,
            },
            'appointmentDateTimeDg': {
                'ctype': 'datetime',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': '',
                'action_type': False,
                'no_clone': False,
                'transform': {'type': 'datetime'},
                'datetime': True,
                'min': None,
                'max': None,
                'time': True,
                'date': True,
            },
        },
        'dataGrid2': {
            'update_datetime': {'transform': {'type': 'datetime'}},
            'create_datetime': {'transform': {'type': 'datetime'}},
            'birthdateDg': {
                'ctype': 'datetime',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': '',
                'action_type': False,
                'no_clone': False,
                'transform': {'type': 'date'},
                'datetime': False,
                'min': None,
                'max': None,
                'time': False,
                'date': True,
            },
            'appointmentDateTimeDg': {
                'ctype': 'datetime',
                'disabled': False,
                'readonly': False,
                'hidden': False,
                'required': False,
                'unique': False,
                'component': 'Component',
                'calculateServer': '',
                'action_type': False,
                'no_clone': False,
                'transform': {'type': 'datetime'},
                'datetime': True,
                'min': None,
                'max': None,
                'time': True,
                'date': True,
            },
        },
    }
    assert test_2.nested_transform_data_value == {
        'dataGrid': {
            'birthdateDg': {'type': 'date'},
            'appointmentDateTimeDg': {'type': 'datetime'},
        },
        'dataGrid2': {
            'birthdateDg': {'type': 'date'},
            'appointmentDateTimeDg': {'type': 'datetime'},
        },
    }
    assert test_2.select_fields == {
        'favouriteSeason': {
            'multi': False,
            'default': None,
            'properties': {},
            'src': 'values',
            'resource_id': '',
        },
        'favouriteFood': {
            'multi': True,
            'default': None,
            'properties': {},
            'src': 'values',
            'resource_id': '',
        },
        'favouriteSeasonDg': {
            'multi': False,
            'default': None,
            'properties': {},
            'src': 'values',
            'resource_id': '',
        },
        'post_id': {
            'default': '',
            'header_key': '',
            'header_value_key': '',
            'multi': False,
            'properties': {'id': 'id', 'label': 'title'},
            'src': 'url',
            'url': 'https://jsonplaceholder.typicode.com/posts',
        },
    }

    assert test_2.select_options == {
        'favouriteSeason': [
            {'label': 'Spring', 'value': 'spring'},
            {'label': 'Summer', 'value': 'summer'},
            {'label': 'Autumn', 'value': 'autumn'},
            {'label': 'Winter', 'value': 'winter'},
        ],
        'favouriteFood': [
            {'label': 'Italian', 'value': 'italian'},
            {'label': 'Mexican', 'value': 'mexican'},
            {'label': 'Chinese', 'value': 'chinese'},
            {'label': 'Fastfood', 'value': 'fastfood'},
        ],
        'favouriteSeasonDg': [
            {'label': 'Spring', 'value': 'spring'},
            {'label': 'Summer', 'value': 'summer'},
            {'label': 'Autumn', 'value': 'autumn'},
            {'label': 'Winter', 'value': 'winter'},
        ],
        'post_id': {},
    }
    assert "rec_name" in test_2.no_clone_field_keys
    test_2.new({"rec_name": "test"})
    assert isinstance(test_2.instance, MainModel) is True
    assert test_2.instance.rec_name == "test"
    # pop appointmentDateTime1 needed for nexts tests
    formio_data_json.pop("appointmentDateTime1")
    test_2.new(formio_data_json)
    assert test_2.instance.textFieldTab1 == "text in tab 1"
    assert test_2.instance.email == 'name@company.it'
    assert len(test_2.instance.dataGrid) == 2
    assert test_2.instance.dataGrid[0].textField == 'abc'
    assert test_2.instance.dataGrid[0].birthdateDg == BasicModel.iso_to_utc(
        "1987-12-17T00:00:00Z"
    )
    assert (
        test_2.instance.dataGrid[0].appointmentDateTimeDg
        == BasicModel.default_datetime()
    )
    assert test_2.instance.dataGrid[1].textField == 'def'
    assert test_2.instance.dataGrid[1].birthdateDg == BasicModel.iso_to_utc(
        "1990-01-01T00:00:00Z"
    )
    assert (
        test_2.instance.dataGrid[1].appointmentDateTimeDg
        == BasicModel.default_datetime()
    )
    assert test_2.instance.dataGrid2 == []
    assert (
        test_2.instance.survey['howWouldYouRateTheFormIoPlatform']
        == 'excellent'
    )


@pytestmark
async def test_make_form_cond_schema():
    schema = await get_formio_schema_conditional()
    formio_data_json = await get_formio_schema_conditional_data_hide()
    test_2 = ModelMaker("component", tz="Europe/Rome")
    test_2.from_formio(schema)
    assert test_2.model_name == "component"
    test_2.new(formio_data_json)
    assert test_2.instance.username == "wrong"
    assert test_2.realted_fields_logic == {
        'username': ['secret'],
        'password': ['secret'],
    }
    d = test_2.instance.get_dict(exclude=["id"])
    assert d == {
        'username': 'wrong',
        'password': 'incorrect',
        'secret': 'Secret message',
        'rec_name': '',
    }


def _field_rule_schema():
    """Schema minimale (stessi campi usati da Component/ModelMaker, vedi
    test_password_fields_encryption_decryption per lo stesso stile
    minimale) con 4 campi che coprono i casi da verificare per l'ACL a
    livello di campo (f_rule/f_rule_cond, layer 3)."""
    return {
        "rec_name": "test_field_rule_model",
        "components": [
            {
                "key": "codicefiscale",
                "type": "textfield",
                "label": "Codice Fiscale",
                "input": True,
                "properties": {
                    "f_rule": {"read": ["gdpr", "dpo"], "write": ["gdpr"]},
                    "f_rule_cond": {
                        "owner_uid": {"$eq": {"var": "user.uid"}}
                    },
                },
            },
            {
                "key": "badRuleField",
                "type": "textfield",
                "label": "Bad Rule",
                "input": True,
                "properties": {
                    # "read" deve essere una lista di stringhe, non una
                    # stringa nuda -> shape non valida, scartata.
                    "f_rule": {"read": "gdpr"},
                    # f_rule_cond deve essere un dict -> shape non valida,
                    # scartata.
                    "f_rule_cond": ["owner_uid", "u1"],
                },
            },
            {
                "key": "plainField",
                "type": "textfield",
                "label": "Plain",
                "input": True,
            },
        ],
    }


def test_field_rule_and_condition_extracted():
    """Campo con f_rule/f_rule_cond validi: entrambi finiscono, verbatim,
    negli accumulatori ModelMaker.field_rules/field_rule_conditions —
    questi alimentano get_field_rules()/get_field_rules_conditions() baked
    a codegen-time da OzonOrm.make_local_model()."""
    schema = _field_rule_schema()
    mm = ModelMaker("test_field_rule_model", tz="Europe/Rome")
    mm.from_formio(schema)

    assert mm.field_rules["codicefiscale"] == {
        "read": ["gdpr", "dpo"],
        "write": ["gdpr"],
    }
    assert mm.field_rule_conditions["codicefiscale"] == {
        "owner_uid": {"$eq": {"var": "user.uid"}}
    }


def test_field_rule_invalid_shape_discarded():
    """Shape non valida (f_rule.read non e' una lista di stringhe,
    f_rule_cond non e' un dict) -> scartata silenziosamente (solo
    logger.warning), il campo NON compare in nessuno dei due accumulatori.
    Fail-closed di proposito: config malformata non deve produrre un
    grant/reveal implicito."""
    schema = _field_rule_schema()
    mm = ModelMaker("test_field_rule_model", tz="Europe/Rome")
    mm.from_formio(schema)

    assert "badRuleField" not in mm.field_rules
    assert "badRuleField" not in mm.field_rule_conditions


def test_field_without_rule_properties_not_restricted():
    """Campo senza f_rule/f_rule_cond nelle properties (baseline, la
    stragrande maggioranza dei campi) non finisce in nessuno dei due
    accumulatori -> get_restricted_fields() (unione delle chiavi) non lo
    includera'."""
    schema = _field_rule_schema()
    mm = ModelMaker("test_field_rule_model", tz="Europe/Rome")
    mm.from_formio(schema)

    assert "plainField" not in mm.field_rules
    assert "plainField" not in mm.field_rule_conditions
    restricted = sorted(set(mm.field_rules) | set(mm.field_rule_conditions))
    assert restricted == ["codicefiscale"]


def test_basic_model_field_rule_fallback_defaults():
    """Un model NON generato da codegen (o generato prima di questa
    feature) deve degradare a liste/dict vuoti invece di AttributeError —
    vedi il default aggiunto in BaseModels.BasicModel, stesso pattern di
    secret_fields()/get_unique_fields()."""
    assert BasicModel.get_restricted_fields() == []
    assert BasicModel.get_field_rules() == {}
    assert BasicModel.get_field_rules_conditions() == {}


def _supervised_todo(key):
    return {
        "type": "fieldset",
        "key": key,
        "properties": {
            "rec_name": "supervised_todo",
            "action_type": "task",
            "admin": "true",
        },
        "components": [
            {
                "type": "button",
                "key": f"btn_{key}",
                "input": True,
            },
            {
                "type": "checkbox",
                "key": f"todo_{key}",
                "label": "Todo",
                "input": True,
                "defaultValue": False,
            },
        ],
    }


def test_supervised_todo_builds_step_actions_and_checkbox_fields():
    schema = {
        "components": [
            _supervised_todo("approval"),
            _supervised_todo("verification"),
        ]
    }
    mm = ModelMaker("test_step_actions", tz="Europe/Rome")
    model = mm.from_formio(schema)

    assert mm.step_actions == {
        "todo_supervised_approval": {
            "rec_name": "supervised_todo",
            "action_type": "task",
            "admin": "true",
            "url_action": "/step/approval",
        },
        "todo_supervised_verification": {
            "rec_name": "supervised_todo",
            "action_type": "task",
            "admin": "true",
            "url_action": "/step/verification",
        },
    }
    assert "todo_approval" in model.model_fields
    assert "todo_verification" in model.model_fields
    assert "btn_approval" not in model.model_fields


def test_regular_fieldset_does_not_build_step_action():
    schema = {
        "components": [
            {
                "type": "fieldset",
                "key": "ordinary",
                "properties": {"rec_name": "ordinary"},
                "components": [],
            }
        ]
    }
    mm = ModelMaker("test_regular_fieldset", tz="Europe/Rome")
    mm.from_formio(schema)

    assert mm.step_actions == {}


def test_step_field_uses_overridden_step_fields():
    class StepModel(BasicModel):
        @classmethod
        def step_fields(cls) -> dict:
            return {"todo_supervised_approval": {"url_action": "/step/approval"}}

    assert StepModel.step_field("todo_supervised_approval") == {
        "url_action": "/step/approval"
    }
    assert StepModel.step_field("missing") is None
