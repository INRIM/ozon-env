from types import SimpleNamespace

import bson
import pytest
from ozonenv.core.BaseModels import BasicModel, User
from ozonenv.core.OzonModel import OzonModelBase
from pydantic import Field


class ObfuscateTestModel(BasicModel):
    # I tre casi che il builder deve distinguere: default esplicito,
    # default_factory (field.default e' PydanticUndefined) e required
    # (idem, ma senza factory da chiamare).
    label: str = ""
    payload: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    seats: int = 0
    code: str


def _projection(model, obfuscate_fields):
    # build_projection_from_obfuscate_fields usa solo `self.model`.
    return OzonModelBase.build_projection_from_obfuscate_fields(
        SimpleNamespace(model=model), obfuscate_fields
    )


def test_default_factory_fields_are_bson_encodable():
    # Regressione: `field.default` e' PydanticUndefined sui campi con
    # default_factory, finiva dentro {"$literal": ...} e pymongo
    # sollevava InvalidDocument su ogni find(obfuscate_fields=[...]) che
    # toccasse un dict/list — es. `user_data` sul model `user`.
    projection = _projection(User, ["user_data", "user", "token"])

    assert projection["user_data"] == {"$literal": {}}
    assert projection["user"] == {"$literal": {}}
    assert projection["token"] == {"$literal": {}}
    # e' questo che sollevava: cannot encode object: PydanticUndefined
    bson.encode({"$project": projection})


def test_obfuscated_values_validate_back_into_the_model():
    # Il record oscurato ripassa da _load_data: un None su un
    # `dict[str, str]` non Optional sarebbe un ValidationError al posto
    # dell'InvalidDocument.
    projection = _projection(
        ObfuscateTestModel, ["label", "payload", "tags", "seats", "code"]
    )
    masked = {
        name: spec["$literal"]
        for name, spec in projection.items()
        if isinstance(spec, dict)
    }

    bson.encode(masked)
    record = ObfuscateTestModel(**masked)

    assert record.label == "**OMISSIS**"
    assert record.payload == {}
    assert record.tags == []
    assert record.seats == 0
    # required senza default: vuoto derivato dall'annotazione
    assert record.code == "**OMISSIS**"


def test_mutable_empties_are_not_shared_between_fields():
    projection = _projection(ObfuscateTestModel, ["payload", "tags"])

    payload = projection["payload"]["$literal"]
    tags = projection["tags"]["$literal"]
    payload["x"] = 1
    tags.append("x")

    assert _projection(ObfuscateTestModel, ["payload"])["payload"] == {
        "$literal": {}
    }
    assert _projection(ObfuscateTestModel, ["tags"])["tags"] == {
        "$literal": []
    }


def test_non_obfuscated_fields_keep_their_reference():
    projection = _projection(ObfuscateTestModel, ["label"])

    assert projection["payload"] == "$payload"
    assert projection["seats"] == "$seats"


def test_explicit_obfuscate_value_wins():
    class WithObfuscateValue(BasicModel):
        secret: dict[str, str] = Field(
            default_factory=dict, json_schema_extra={"obfuscate_value": "***"}
        )

    projection = _projection(WithObfuscateValue, ["secret"])

    assert projection["secret"] == {"$literal": "***"}


@pytest.mark.parametrize("field_name", sorted(User.model_fields))
def test_every_user_field_is_encodable_when_obfuscated(field_name):
    # Il model `user` e' quello colpito in produzione: nessun campo deve
    # produrre un literal non encodabile.
    bson.encode({"$project": _projection(User, [field_name])})


def test_identity_fields_are_never_obfuscated_in_pipeline():
    # `id` ha default_factory=PyObjectId: mascherarlo non nasconderebbe
    # nulla, fabbricherebbe un ObjectId NUOVO — il record tornerebbe con
    # un'identita' inventata e ogni update/by_id successivo lavorerebbe su
    # un id inesistente.
    projection = _projection(User, ["id", "uid"])

    assert projection["id"] == "$id"
    assert projection["uid"] == {"$literal": "**OMISSIS**"}


def test_user_masked_record_validates_back():
    # Il model colpito in produzione: ogni campo mascherabile deve
    # rivalidare in User, non solo essere BSON-encodable.
    maskable = [
        name for name in User.model_fields if name not in ("id", "_id")
    ]
    projection = _projection(User, maskable)
    masked = {
        name: spec["$literal"]
        for name, spec in projection.items()
        if isinstance(spec, dict)
    }

    bson.encode(masked)
    record = User(**masked)

    assert record.uid == "**OMISSIS**"
    assert record.user_data == {}
    assert record.groups == []
    assert record.last_update == 0
