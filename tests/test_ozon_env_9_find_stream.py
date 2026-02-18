from ozonenv.OzonEnv import OzonEnv
from test_common import *

pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_stream_and_find_large_dataset():

    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()

    model = env.get("test_form_1")

    # -----------------------------
    # Insert 200 records
    # -----------------------------
    for i in range(200):
        rec = await model.new({
            "post_id": str(i),
            "firstName": f"user_{i}"
        })
        await model.insert(rec)

    # -----------------------------
    # Test find() list mode
    # -----------------------------
    records = await model.find({})
    assert isinstance(records, list)
    assert len(records) >= 200

    # -----------------------------
    # Test stream_find()
    # -----------------------------
    count = 0
    async for rec in model.stream_find({}):
        assert rec is not None
        count += 1

    assert count == len(records)

    # -----------------------------
    # Test limit
    # -----------------------------
    limited = await model.find({}, limit=10)
    assert len(limited) == 10

    stream_count = 0
    async for _ in model.stream_find({}, limit=15):
        stream_count += 1

    assert stream_count == 15

    # -----------------------------
    # Test consistency stream vs list
    # -----------------------------
    list_ids = [r.rec_name for r in records]

    stream_ids = []
    async for r in model.stream_find({}):
        stream_ids.append(r.rec_name)

    assert set(list_ids) == set(stream_ids)

    await env.close_env()

@pytest.mark.asyncio
async def test_stream_memory_stable():

    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()

    model = env.get("test_form_1")

    for i in range(500):
        rec = await model.new({
            "post_id": str(i),
            "firstName": f"user_{i}"
        })
        await model.insert(rec)

    count = 0
    async for _ in model.stream_find({}):
        count += 1

    assert count >= 500

    await env.close_env()


import pytest
import uuid


@pytest.mark.asyncio
async def test_obfuscate_multiple_fields():

    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()

    model = env.get("test_form_1")

    unique_name = f"Mario_{uuid.uuid4().hex}"

    # Inserisco record reale
    rec = await model.new({
        "firstName": unique_name,
        "email": "mario.rossi@test.com",
        "birthdate": "1990-01-01",
        "howManySeats": 4,
        "favouriteSeason": "summer",
        "survey": {
            "overallExperience": "excellent"
        }
    })

    rec = await model.insert(rec)

    # -------------------------
    # 1) Senza obfuscation (filtrato per _id)
    # -------------------------
    records = await model.find({"id": rec.id})
    assert len(records) == 1

    r = records[0]

    assert r.firstName == unique_name
    assert r.email == "mario.rossi@test.com"
    assert r.howManySeats == 4
    assert r.favouriteSeason == "summer"

    # -------------------------
    # 2) Con obfuscation
    # -------------------------
    records_obf = await model.find(
        {"id": rec.id},
        obfuscate_fields=[
            "firstName",
            "email",
            "birthdate",
            "howManySeats",
            "favouriteSeason",
            "survey",
        ]
    )

    assert len(records_obf) == 1
    r_obf = records_obf[0]

    # Verifica che siano cambiati
    assert r_obf.firstName != unique_name
    assert r_obf.email != "mario.rossi@test.com"
    assert r_obf.howManySeats != 4
    assert r_obf.favouriteSeason != "summer"

    # Verifica che campi strutturali NON siano alterati
    assert r_obf.data_model == r.data_model
    assert r_obf.rec_name == r.rec_name
    assert r_obf.id == r.id

    await env.close_env()

@pytest.mark.asyncio
async def test_stream_obfuscate_fields():

    env = OzonEnv()
    await env.init_env()
    env.params = {"current_session_token": "BA6BA930"}
    await env.session_app()

    model = env.get("test_form_1")

    unique_name = f"Mario_{uuid.uuid4().hex}"

    rec = await model.new({
        "firstName": unique_name,
        "email": "mario.rossi@test.com",
        "birthdate": "1990-01-01",
        "howManySeats": 4,
        "favouriteSeason": "summer",
        "survey": {
            "overallExperience": "excellent"
        }
    })
    rec = await model.insert(rec)

    found = False

    async for rec_stream in model.stream_find(
        {"id": rec.id},
        obfuscate_fields=["firstName", "email","survey"]
    ):
        found = True
        assert rec_stream.firstName != unique_name
        assert rec_stream.email == "**OMISSIS**"
        assert rec_stream.id == rec.id
        assert rec_stream.favouriteSeason == "summer"
        assert rec_stream.survey == {}
        break

    assert found is True

    await env.close_env()
