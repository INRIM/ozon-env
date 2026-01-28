from ozonenv.OzonEnv import OzonEnv
from test_common import *

pytestmark = pytest.mark.asyncio


@pytestmark
async def test_local_transaction():
    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()
    env.local_transaction_start()
    test_form_1_model = env.get('test_form_1')

    # test rollback insert
    forms = await test_form_1_model.find({})
    assert len(forms) == 4

    record = await test_form_1_model.new(
        {"post_id": "4", "firstName": "test1"}
    )
    record = await test_form_1_model.insert(record)

    forms = await test_form_1_model.find({})
    assert len(forms) == 5
    await env.local_transaction_rollback()

    forms = await test_form_1_model.find({})
    assert len(forms) == 4

    # test rollback delete
    env.local_transaction_start()
    delete_rec_name = forms[3].rec_name

    await test_form_1_model.remove(forms[3])
    forms = await test_form_1_model.find({})
    assert len(forms) == 3
    await env.local_transaction_rollback()
    forms = await test_form_1_model.find({})
    assert len(forms) == 4
    assert forms[3].rec_name == delete_rec_name

    # test rollback insert and update
    env.local_transaction_start()
    record = await test_form_1_model.new(
        {"post_id": "5", "firstName": "test2"}
    )

    record = await test_form_1_model.insert(record)
    assert record.post_id == "5"
    assert record.get("data_value.post_id") == ""

    record.selection_value(
        "post_id",
        "6",
        "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
    )
    record = await test_form_1_model.update(record)

    assert record.post_id == "6"
    assert (
        record.get("data_value.post_id")
        == "sunt aut facere repellat provident occaecati excepturi optio reprehenderit"
    )

    await env.local_transaction_rollback()

    recordr = await test_form_1_model.by_name(record.rec_name)
    assert recordr is None

    # test rollback update
    record = await test_form_1_model.new(
        {"post_id": "5", "firstName": "test2"}
    )
    record = await test_form_1_model.insert(record)
    assert record.post_id == "5"
    assert record.get("data_value.post_id") == ""

    env.local_transaction_start()
    record.selection_value(
        "post_id",
        "6",
        "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
    )
    record = await test_form_1_model.update(record)

    assert record.post_id == "6"
    assert (
        record.get("data_value.post_id")
        == "sunt aut facere repellat provident occaecati excepturi optio reprehenderit"
    )

    await env.local_transaction_rollback()

    recordr2 = await test_form_1_model.by_name(record.rec_name)
    assert recordr2.rec_name == record.rec_name
    assert recordr2.post_id == "5"
    assert recordr2.get("data_value.post_id") == ""

    await env.close_env()
