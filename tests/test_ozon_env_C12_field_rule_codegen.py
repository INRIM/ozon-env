import pytest
from ozonenv.OzonEnv import OzonEnv
from test_common import *

pytestmark = pytest.mark.asyncio


@pytestmark
async def test_field_rule_classmethods_baked_by_codegen():
    """End-to-end (component save -> real datamodel-codegen subprocess ->
    generated .py) equivalente a test_password_fields_encryption_decryption
    per secret_fields(), ma per get_field_rules()/get_field_rules_
    conditions()/get_restricted_fields() (layer 3 field-level ACL)."""
    env = OzonEnv()
    await auth_env(env)

    schema = {
        "rec_name": "test_field_rule_model",
        "title": "Test Field Rule Model",
        "name": "test_field_rule_model",
        "components": [
            {
                "key": "name",
                "type": "textfield",
                "label": "Name",
            },
            {
                "key": "codicefiscale",
                "type": "textfield",
                "label": "Codice Fiscale",
                "properties": {
                    "f_rule": {
                        "read": ["gdpr", "dpo"],
                        "write": ["gdpr"],
                    },
                    "f_rule_cond": {
                        "owner_uid": {"$eq": {"var": "user.uid"}}
                    },
                },
            },
        ],
    }

    coll = env.db.engine.get_collection("test_field_rule_model")
    await coll.delete_many({})

    component = await env.insert_update_component(schema)
    assert component.rec_name == "test_field_rule_model"

    model = await env.add_model("test_field_rule_model")
    assert model.name == "test_field_rule_model"

    assert hasattr(model.model, "get_field_rules")
    assert hasattr(model.model, "get_field_rules_conditions")
    assert hasattr(model.model, "get_restricted_fields")

    assert model.model.get_field_rules() == {
        "codicefiscale": {"read": ["gdpr", "dpo"], "write": ["gdpr"]}
    }
    assert model.model.get_field_rules_conditions() == {
        "codicefiscale": {"owner_uid": {"$eq": {"var": "user.uid"}}}
    }
    assert model.model.get_restricted_fields() == ["codicefiscale"]
    # campo senza f_rule/f_rule_cond non deve comparire da nessuna parte.
    assert "name" not in model.model.get_field_rules()
    assert "name" not in model.model.get_field_rules_conditions()

    await env.close_db()


@pytestmark
async def test_field_rule_updated_on_component_regeneration():
    """Trigger reale nel workflow: componente gia' esistente, editato per
    AGGIUNGERE f_rule a un campo -> insert_update_component prende il ramo
    update (OzonOrm.update_model -> stessa init_model_and_write_code_from_
    schema/make_local_model del create) e il modello rigenerato deve
    riflettere la nuova regola, non quella (assente) di prima."""
    env = OzonEnv()
    await auth_env(env)

    base_schema = {
        "rec_name": "test_field_rule_regen_model",
        "title": "Test Field Rule Regen Model",
        "name": "test_field_rule_regen_model",
        "components": [
            {"key": "name", "type": "textfield", "label": "Name"},
            {"key": "codicefiscale", "type": "textfield", "label": "CF"},
        ],
    }

    coll = env.db.engine.get_collection("test_field_rule_regen_model")
    await coll.delete_many({})

    await env.insert_update_component(base_schema)
    model = await env.add_model("test_field_rule_regen_model")
    assert model.model.get_field_rules() == {}
    assert model.model.get_restricted_fields() == []

    updated_schema = {
        **base_schema,
        "components": [
            {"key": "name", "type": "textfield", "label": "Name"},
            {
                "key": "codicefiscale",
                "type": "textfield",
                "label": "CF",
                "properties": {
                    "f_rule": {"read": ["gdpr"], "write": ["gdpr"]},
                },
            },
        ],
    }
    await env.insert_update_component(updated_schema)
    model = await env.add_model("test_field_rule_regen_model")

    assert model.model.get_field_rules() == {
        "codicefiscale": {"read": ["gdpr"], "write": ["gdpr"]}
    }
    assert model.model.get_restricted_fields() == ["codicefiscale"]

    await env.close_db()
