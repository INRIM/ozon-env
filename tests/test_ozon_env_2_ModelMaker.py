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
    test_2 = ModelMaker("component",tz="Europe/Rome")
    test_2.from_formio(schema)
    assert test_2.model_name == "component"
    assert isinstance(test_2.model, ModelMetaclass) is True
    assert test_2.unique_fields == ["rec_name", "firstName"]
    assert test_2.required_fields == ["rec_name", "firstName"]
    assert test_2.components_logic == []
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
    assert test_2.instance.dataGrid[1].textField == 'def'
    assert test_2.instance.survey[
               'howWouldYouRateTheFormIoPlatform'] == 'excellent'


@pytestmark
async def test_make_form_cond_schema():
    schema = await get_formio_schema_conditional()
    formio_data_json = await get_formio_schema_conditional_data_hide()
    test_2 = ModelMaker("component", tz="Europe/Rome")
    test_2.from_formio(schema)
    assert test_2.model_name == "component"
    test_2.new(formio_data_json)
    assert test_2.instance.username == "wrong"
    assert test_2.realted_fields_logic == {'username': ['secret'],
                                           'password': ['secret']}
    d = test_2.instance.get_dict(exclude=["id"])
    assert d == {'username': 'wrong', 'password': 'incorrect',
                 'secret': 'Secret message', 'rec_name': ''}

