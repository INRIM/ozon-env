from ozonenv.OzonEnv import OzonEnv
from test_common import *

pytestmark = pytest.mark.asyncio


@pytestmark
async def test_riga_doc_select_model_distinct():
    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()
    riga_doc_model = env.get('riga_doc')

    record = await riga_doc_model.new({
        "parent": "DOC99999"
    })

    assert record.parent == "DOC99999"
    assert record.get("data_value.parent") == "8 - 2022"

    record = await riga_doc_model.insert(record)

    assert record.parent == "DOC99999"
    assert record.get("data_value.parent") == "8 - 2022"

    record.selection_value("parent", "DOCNOTEXISTS", "NOT EXISTS")
    record = await riga_doc_model.update(record)

    assert record.parent == "DOCNOTEXISTS"
    assert record.get("data_value.parent") == "NOT EXISTS"

    await env.close_env()

@pytestmark
async def test_riga_doc_select_url():
    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()
    test_form_1_model = env.get('test_form_1')

    record = await test_form_1_model.new({
        "post_id": "3"
    })

    assert record.post_id == "3"
    assert record.get("data_value.post_id") == ""

    record = await test_form_1_model.insert(record)

    assert record.post_id == "3"
    assert record.get("data_value.post_id") == ""

    record.selection_value("post_id", "1", "sunt aut facere repellat provident occaecati excepturi optio reprehenderit")
    record = await test_form_1_model.update(record)

    assert record.post_id == "1"
    assert record.get("data_value.post_id") == "sunt aut facere repellat provident occaecati excepturi optio reprehenderit"

    await env.close_env()