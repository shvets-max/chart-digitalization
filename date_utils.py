import re

from dateutil.parser import parse


class DateComponentClassifier:
    MONTHS = {
        m.lower()
        for m in [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "January",
            "February",
            "March",
            "April",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    }

    def classify(self, s):
        s = s.strip().replace("‘", "").replace("’", "")
        # Year: 4 digits or 2 digits (assume 2000+)
        if re.fullmatch(r"\d{4}", s):
            return "year"
        if re.fullmatch(r"\d{2}", s) and int(s) > 31:
            return "year"
        # Month: text or 1-2 digit number 1-12
        if s.lower() in self.MONTHS or re.fullmatch(r"0?[1-9]|1[0-2]", s):
            return "month"
        # Day: 1-2 digit number 1-31
        if re.fullmatch(r"0?[1-9]|[12][0-9]|3[01]", s):
            return "day"
        # Year and month: e.g. 2023-05, 05/2023, 2023.05, May 2023, 05.23, 05-23
        if re.fullmatch(r"\d{4}[-/.]\d{1,2}", s) or re.fullmatch(
            r"\d{1,2}[-/.]\d{4}", s
        ):
            return "year and month"
        if re.fullmatch(r"\d{1,2}[-/.]\d{2}", s):
            return "year and month"
        if any(month in s.lower() for month in self.MONTHS) and re.search(r"\d{4}", s):
            return "year and month"
        # Month and day: e.g. 12-25, Dec 25, 25 Dec
        if re.fullmatch(r"\d{1,2}[-/.]\d{1,2}", s):
            return "month and day"
        if any(month in s.lower() for month in self.MONTHS) and re.search(
            r"\d{1,2}", s
        ):
            return "month and day"
        # Day, month, and year: e.g. 25-12-2023, 2023/12/25, 25 Dec 2023
        try:
            # dt = parse(s, fuzzy=False)
            return "day, month and year"
        except Exception:
            pass
        return "unknown"
