# Copyright INRIM (https://www.inrim.eu)
# See LICENSE file for full licensing details.

import datetime
import locale
from collections import namedtuple
from datetime import date, datetime, timedelta, time
from zoneinfo import ZoneInfo

from dateutil.parser import parse

try:
    locale.setlocale(locale.LC_ALL, "it_IT")
except:
    pass


class DateEngine:
    def __init__(
        self,
        UI_DATE_MASK="%d/%m/%Y",
        REPORT_DATE_MASK="%d-%m-%Y",
        UI_DATETIME_MASK="%d/%m/%Y %H:%M:%S",
        TZ="Europe/Rome",
    ):
        super().__init__()
        self.client_date_mask = UI_DATE_MASK
        self.client_datetime_mask = UI_DATETIME_MASK
        self.report_date_mask = REPORT_DATE_MASK
        self.tz = TZ

    @property
    def _today(self) -> datetime:
        dt = datetime.combine(date.today(), time.min)
        # Imposta il timezone desiderato
        return dt.replace(tzinfo=ZoneInfo(self.tz))

    @property
    def curr_year(self) -> int:
        return date.today().year

    @property
    def curr_month(self) -> int:
        return date.today().month

    @property
    def curr_day(self) -> int:
        return date.today().day

    @property
    def today(self) -> str:
        return self._today.isoformat()

    @property
    def today_ui(
        self,
    ):
        return self._today.strftime(self.client_date_mask)

    @property
    def todaymax(self) -> datetime:
        return datetime.combine(self._today, time.max)

    def parse_to_utc_datetime(self, dt_str):
        value = parse(dt_str)
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo(self.tz))
        return value.astimezone(ZoneInfo("UTC"))

    def year_range(self, year=0, datetime_o=datetime) -> dict:
        if year == 0:
            year = self._today.year
        return {
            "date_from": datetime_o.min.replace(year=year),
            "date_to": datetime_o.max.replace(year=year),
        }

    def month_range(self, year=0, month=0, datetime_o=datetime) -> dict:
        if year == 0:
            year = self._today.year
        if month == 0:
            month = self._today.month
        return {
            "date_from": datetime_o.min.replace(year=year, month=month),
            "date_to": datetime_o.max.replace(year=year, month=month),
        }

    def get_date_delta_from_today(self, deltat) -> datetime:
        return self._today + timedelta(deltat)

    def gen_date_delta_from_today_ui(self, deltat):
        return (self.get_date_delta_from_today(deltat)).strftime(
            self.client_datetime_mask
        )

    def gen_date_min_max_ui(
        self, min_day_delata_date_from=1, max_day_delata_date_to=5
    ):
        min = self.gen_date_delta_from_today_ui(min_day_delata_date_from)
        max = self.gen_date_delta_from_today_ui(max_day_delata_date_to)
        return min, max

    def gen_date_from_to_gui_dict(
        self, min_day_delata_date_from=1, max_day_delata_date_to=5
    ) -> dict:
        dtmin, dtmax = self.gen_date_min_max_ui(
            min_day_delata_date_from, max_day_delata_date_to
        )
        res = {
            "date_from": dtmin,
            "date_to": dtmax,
        }
        return res

    def gen_date_from_to_dict(
        self, min_day_delata_date_from=1, max_day_delata_date_to=5
    ):
        dtmin, dtmax = self.gen_date_min_max(
            min_day_delata_date_from, max_day_delata_date_to
        )
        res = {
            "date_from": dtmin,
            "date_to": dtmax,
        }
        return res

    def is_today_or_less(self, date_test) -> bool:
        return date_test <= date.today()

    def is_less_today(self, date_test) -> bool:
        return date_test < self._today

    def check_dates_overlap(self, list_date1, list_date2) -> bool:
        Range = namedtuple("Range", ["start", "end"])
        r1 = Range(start=list_date1[0], end=list_date1[1])
        r2 = Range(start=list_date2[0], end=list_date2[1])
        latest_start = max(r1.start, r2.start)
        earliest_end = min(r1.end, r2.end)
        delta = (earliest_end - latest_start).days + 1
        overlap = max(0, delta)
        return overlap

    def format_in_client_tz(
        self, date_to_parse: str, dt_type: str = "datetime"
    ) -> str:
        # date_to_parse: stringa ISO con offset o “naive” (meglio con offset)
        # client_tz: es: "Europe/Rome"
        # client_mask: es: "%Y-%m-%dT%H:%M:%S%z" o altro formato compatibile

        # 1. Parsing: da ISO (da stringa con offset)
        # può produrre aware datetime se la stringa ha offset
        dt = datetime.fromisoformat(date_to_parse)

        # 2. Conversione nella timezone client
        if dt_type == "datetime":
            dt = dt.replace(tzinfo=ZoneInfo(self.tz))

            dt_client = dt.astimezone(ZoneInfo(self.tz))
        else:
            dt_client = dt

        # 3. Formattazione
        if dt_type == "datetime":
            out = dt_client.strftime(self.client_datetime_mask)
        else:
            out = dt_client.strftime(self.client_date_mask)
        return out

    def to_ui(self, date_obj, dt_type: str = "datetime") -> str:
        if isinstance(date_obj, str):
            return self.format_in_client_tz(date_obj, dt_type)
        if date_obj.tzinfo is None or date_obj.tzinfo != ZoneInfo(self.tz):
            date_obj = date_obj.astimezone(ZoneInfo(self.tz))
        if dt_type == "datetime":
            return date_obj.strftime(self.client_datetime_mask)
        else:
            return date_obj.strftime(self.client_date_mask)
