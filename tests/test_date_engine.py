# Copyright INRIM (https://www.inrim.eu)
# See LICENSE file for full licensing details.
"""Test puri su DateEngine: nessun mongo, nessun keycloak."""

from datetime import datetime
from zoneinfo import ZoneInfo

from ozonenv.core.DateEngine import DateEngine


def make_engine() -> DateEngine:
    return DateEngine(
        UI_DATE_MASK="%d/%m/%Y",
        UI_DATETIME_MASK="%d/%m/%Y %H:%M:%S",
        TZ="Europe/Rome",
    )


def test_to_ui_none_returns_empty_string():
    """Un campo datetime presente ma a null arriva qui come None.

    Succede sui dati migrati da versioni precedenti: il chiamante fa
    `input_data.get(name, defaultdt)` e il default scatta solo se la chiave
    manca. Prima del guard questo alzava AttributeError su .tzinfo e faceva
    fallire l'init dei modelli.
    """
    dte = make_engine()
    assert dte.to_ui(None) == ""
    assert dte.to_ui(None, "datetime") == ""
    assert dte.to_ui(None, "date") == ""


def test_to_ui_aware_datetime():
    dte = make_engine()
    dt = datetime(2026, 9, 1, 10, 30, 0, tzinfo=ZoneInfo("Europe/Rome"))
    assert dte.to_ui(dt, "datetime") == "01/09/2026 10:30:00"


def test_to_ui_string_is_parsed():
    dte = make_engine()
    out = dte.to_ui("2026-09-01T10:30:00+02:00", "datetime")
    assert out == "01/09/2026 10:30:00"
