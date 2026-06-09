import pytest
from ozonenv.OzonEnv import OzonEnv
from test_common import *

pytestmark = pytest.mark.asyncio


@pytestmark
async def test_password_fields_encryption_decryption():
    env = OzonEnv()
    await auth_env(env)

    # 1. Define schema with a password field
    schema = {
        "rec_name": "test_password_model",
        "title": "Test Password Model",
        "name": "test_password_model",
        "components": [
            {
                "key": "name",
                "type": "textfield",
                "label": "Name"
            },
            {
                "key": "password_field",
                "type": "password",
                "label": "Secret Password"
            }
        ]
    }

    # Clean existing models/data first
    coll = env.db.engine.get_collection("test_password_model")
    await coll.delete_many({})

    # 2. Insert/update component
    component = await env.insert_update_component(schema)
    assert component.rec_name == "test_password_model"

    # 3. Add model (this generates the local model via make_local_model)
    model = await env.add_model("test_password_model")
    assert model.name == "test_password_model"

    # 4. Verify secret_fields classmethod returns the password field
    assert hasattr(model.model, "secret_fields")
    secret_fields_list = model.model.secret_fields()
    assert "password_field" in secret_fields_list

    # 5. Insert a record with a password
    plain_password = "my_secret_pass_123"
    record = await model.new({
        "rec_name": "pass_record_1",
        "name": "Test User",
        "password_field": plain_password
    })

    saved_record = await model.insert(record)
    assert saved_record is not None
    assert saved_record.password_field == plain_password

    # 6. Verify database contains encrypted password
    raw_doc = await coll.find_one({"rec_name": "pass_record_1"})
    assert raw_doc is not None
    assert raw_doc["password_field"] != plain_password
    assert len(raw_doc["password_field"]) > len(plain_password)  # encryption makes it longer usually (Fernet)

    # 7. Verify load reads the decrypted password
    loaded_record = await model.load({"rec_name": "pass_record_1"})
    assert loaded_record is not None
    assert loaded_record.password_field == plain_password

    # 8. Update name and verify password stays encrypted in DB and decrypted in Python
    loaded_record.name = "Updated User Name"
    updated_record = await model.update(loaded_record)
    assert updated_record is not None
    assert updated_record.name == "Updated User Name"
    assert updated_record.password_field == plain_password

    # Reload from DB raw and verify it's still encrypted
    raw_doc_after_update = await coll.find_one({"rec_name": "pass_record_1"})
    assert raw_doc_after_update["password_field"] != plain_password

    # Reload from model and verify it's decrypted
    reloaded_record = await model.load({"rec_name": "pass_record_1"})
    assert reloaded_record.password_field == plain_password

    await env.close_db()
