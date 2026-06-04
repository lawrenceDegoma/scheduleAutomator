"""
data_generator.py

Generates synthetic but realistic training data for:
  1. Demand forecasting (sales, transactions, required headcount)
  2. Barista preferred-hours learning (preference model)

Employee schema mirrors real coffee-shop partner data:
  - partner_id   : unique numeric ID (7 digits, e.g. "1042381")
  - name         : first + last name
  - role         : Barista | Shift Supervisor | Assistant Store Manager | Store Manager
  - All baristas carry a full station skill set (bar, register, drive_thru,
    warming, floor) because partners rotate stations during every shift.

Schedule history schema mirrors real scheduling-system exports:
  - No role column — real exports just show who is scheduled and when.
  - Station assignment happens at shift time by the supervisor on duty.

This module produces data in the same schema that real operational data
will follow so that the model pipeline can be swapped to live data
with zero code changes — only a different DataLoader is required.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import holidays as holiday_lib
import random

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Fake name pools (realistic but not real)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Aisha", "Alex", "Amber", "Andre", "Ashley", "Brianna", "Carlos",
    "Carmen", "Chris", "Chloe", "Danielle", "David", "Diana", "Dylan",
    "Elena", "Emily", "Eric", "Eva", "Gabrielle", "Hannah", "Isaiah",
    "Jacob", "Jade", "James", "Jasmine", "Jessica", "Jordan", "Julia",
    "Justin", "Kayla", "Kevin", "Kylie", "Laura", "Lauren", "Lena",
    "Logan", "Luis", "Madison", "Marcus", "Maria", "Maya", "Michael",
    "Miguel", "Mia", "Natalie", "Nathan", "Nicole", "Noah", "Olivia",
    "Omar", "Patricia", "Rachel", "Riley", "Ryan", "Samantha", "Sara",
    "Sebastian", "Sofia", "Stephanie", "Taylor", "Tiana", "Tyler",
    "Vanessa", "Victoria", "Wesley", "Zoe",
]
LAST_NAMES = [
    "Anderson", "Baker", "Brown", "Campbell", "Carter", "Chen", "Clark",
    "Collins", "Davis", "Diaz", "Evans", "Flores", "Garcia", "Gonzalez",
    "Green", "Hall", "Harris", "Hernandez", "Hill", "Jackson", "Johnson",
    "Jones", "Kim", "King", "Lee", "Lewis", "Lopez", "Martin", "Martinez",
    "Miller", "Mitchell", "Moore", "Morales", "Murphy", "Nguyen", "Patel",
    "Perez", "Phillips", "Ramirez", "Reyes", "Rivera", "Roberts", "Robinson",
    "Rodriguez", "Sanchez", "Scott", "Smith", "Taylor", "Thomas", "Thompson",
    "Torres", "Turner", "Walker", "White", "Williams", "Wilson", "Young",
]


# ===========================================================================
# 1. DEMAND DATA GENERATOR
# ===========================================================================

class DemandDataGenerator:
    """
    Simulates two years of 30-minute-interval sales and transaction data
    for a single coffee shop location.

    Patterns embedded:
      - Hour-of-day curve (morning rush, lunch bump, afternoon lull)
      - Day-of-week variation (Mon–Fri business, Sat–Sun leisure patterns)
      - Seasonal trend (spring/fall slightly higher than summer/winter)
      - Holiday suppression (lower traffic on major US holidays)
      - Promotional lift (random promotion days with +15–30% bump)
      - Weather impact (rain slightly boosts hot-drink sales)
      - Gaussian noise
    """

    OPEN_HOUR = 5      # 5:00 AM
    CLOSE_HOUR = 22    # 10:00 PM (last interval starts at 21:30)
    INTERVAL_MINUTES = 30

    # Base transaction volumes per 30-min interval (7-day x 34-interval shape)
    # Index 0 = Monday, 6 = Sunday
    # Values are "average transactions per 30-min slot" at peak
    DOW_BASE = {
        0: 28,   # Monday
        1: 30,   # Tuesday
        2: 32,   # Wednesday
        3: 31,   # Thursday
        4: 35,   # Friday
        5: 42,   # Saturday
        6: 38,   # Sunday
    }

    def __init__(self, start_date: str = "2024-01-01",
                 end_date: str = "2025-12-31",
                 location_id: str = "LOC001"):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.location_id = location_id
        self.us_holidays = holiday_lib.US(years=range(
            self.start_date.year, self.end_date.year + 1
        ))

    # ------------------------------------------------------------------
    # Hour-of-day transaction shape (sum ≈ 1.0 over operating hours)
    # ------------------------------------------------------------------
    def _hour_shape(self, hour: float) -> float:
        """Return relative demand weight for a given hour (0–1 scale)."""
        # Morning rush: peak at 8am
        morning = np.exp(-0.5 * ((hour - 8.0) / 1.2) ** 2) * 1.0
        # Lunch bump: smaller peak at 12pm
        lunch = np.exp(-0.5 * ((hour - 12.0) / 1.0) ** 2) * 0.35
        # Afternoon: small bump at 3pm (iced drinks)
        afternoon = np.exp(-0.5 * ((hour - 15.0) / 1.5) ** 2) * 0.25
        raw = morning + lunch + afternoon
        return max(raw, 0.02)   # floor — never truly zero during open hours

    # ------------------------------------------------------------------
    # Seasonal multiplier
    # ------------------------------------------------------------------
    def _seasonal_mult(self, month: int) -> float:
        # Peak: March–May (spring), Sept–Nov (fall pumpkin season)
        seasonal = {
            1: 0.88, 2: 0.90, 3: 1.05, 4: 1.10, 5: 1.08,
            6: 0.95, 7: 0.92, 8: 0.94, 9: 1.06, 10: 1.12,
            11: 1.05, 12: 0.97,
        }
        return seasonal.get(month, 1.0)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self) -> pd.DataFrame:
        records = []
        current = self.start_date.date()
        end = self.end_date.date()

        # Pre-generate a promotion calendar: ~10% of days have a promo
        all_days = pd.date_range(self.start_date, self.end_date, freq="D")
        promo_days = set(
            pd.Timestamp(d).date()
            for d in rng.choice(all_days, size=int(len(all_days) * 0.10), replace=False)
        )

        while current <= end:
            dow = current.weekday()
            is_holiday = current in self.us_holidays
            is_promo = current in promo_days
            month = current.month
            seasonal = self._seasonal_mult(month)
            holiday_mult = 0.45 if is_holiday else 1.0
            promo_mult = rng.uniform(1.15, 1.30) if is_promo else 1.0

            # Simple weather: ~20% of days are "rainy" — slight boost
            is_rain = rng.random() < 0.20
            weather_mult = rng.uniform(1.03, 1.08) if is_rain else 1.0

            base = self.DOW_BASE[dow]

            for hour in range(self.OPEN_HOUR, self.CLOSE_HOUR):
                for minute in [0, 30]:
                    slot_hour = hour + minute / 60.0
                    shape = self._hour_shape(slot_hour)
                    mean_tx = (
                        base * shape * seasonal * holiday_mult
                        * promo_mult * weather_mult
                    )
                    transactions = max(0, int(rng.normal(mean_tx, mean_tx * 0.15)))
                    avg_ticket = rng.uniform(6.50, 9.50)
                    sales = round(transactions * avg_ticket, 2)

                    records.append({
                        "location_id": self.location_id,
                        "date": current,
                        "datetime": datetime.combine(
                            current, datetime.min.time()
                        ).replace(hour=hour, minute=minute),
                        "hour": hour,
                        "minute": minute,
                        "day_of_week": dow,
                        "month": month,
                        "is_holiday": int(is_holiday),
                        "is_promotion": int(is_promo),
                        "is_rain": int(is_rain),
                        "transactions": transactions,
                        "sales": sales,
                    })
            current += timedelta(days=1)

        df = pd.DataFrame(records)
        return df


# ===========================================================================
# 2. EMPLOYEE & SCHEDULE DATA GENERATOR
# ===========================================================================

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------
# Real stores have exactly 1 Store Manager, 0–1 Assistant Store Manager,
# 2–4 Shift Supervisors, and the remainder are Baristas.
# All partners (including supervisors) are trained as Baristas — they rotate
# through every station (bar, register, drive-thru, warming, floor) during
# each shift. Role only affects who is the shift leader.
# ---------------------------------------------------------------------------

ROLES = ["Store Manager", "Assistant Store Manager", "Shift Supervisor", "Barista"]

# Every partner can work every station — role is a leadership designation only.
BARISTA_SKILLS = ["bar", "register", "drive_thru", "warming", "floor"]

PREFERENCE_LABELS = ["morning", "midday", "evening", "flexible"]


def _unique_partner_id(existing: set) -> str:
    """Generate a unique 7-digit partner ID that doesn't collide."""
    while True:
        pid = str(random.randint(1000000, 9999999))
        if pid not in existing:
            existing.add(pid)
            return pid


def _random_name(used: set) -> str:
    """Generate a unique first + last name combination."""
    for _ in range(200):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name
    # Fallback: append a number if pool is exhausted
    base = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    return f"{base} {random.randint(2, 99)}"


class EmployeeScheduleGenerator:
    """
    Generates a synthetic partner roster and 2 years of shift history.

    Roster composition per location (matches real store staffing):
      - 1  Store Manager         (salaried, not scheduled via barista system)
      - 0–1 Assistant Store Manager (salaried, light scheduling presence)
      - 2–4 Shift Supervisors    (hourly, full-time, lead every shift)
      - Remainder: Baristas      (hourly, mix of full-time and part-time)

    All partners share the same skill set (bar, register, drive_thru,
    warming, floor) because cross-training is required for employment.
    Role only determines whether a partner can be the shift leader.

    Schedule history schema (mirrors real export):
      partner_id, name, date, day_of_week, shift_start_hour, shift_end_hour,
      scheduled_hours, no_show, manager_override, week_start,
      weekly_hours_accumulated
      — NO role column, matching real scheduling-system CSV exports.
    """

    SHIFT_TYPES = {
        "open":     (5,  13),   # 5 AM – 1 PM  (8 hr)
        "mid":      (9,  17),   # 9 AM – 5 PM  (8 hr)
        "close":    (14, 22),   # 2 PM – 10 PM (8 hr)
        "short_am": (7,  12),   # 7 AM – 12 PM (5 hr)
        "short_pm": (15, 20),   # 3 PM – 8 PM  (5 hr)
    }

    # Store/Assistant Managers rarely appear on the hourly schedule;
    # when they do it's almost always an open or mid shift.
    MANAGER_SHIFT_TYPES = {
        "open": (5, 13),
        "mid":  (9, 17),
    }

    def __init__(self, n_baristas: int = 20,
                 n_supervisors: int = 3,
                 include_asm: bool = True,
                 start_date: str = "2024-01-01",
                 end_date: str = "2025-12-31",
                 location_id: str = "LOC001"):
        self.n_baristas    = n_baristas
        self.n_supervisors = n_supervisors   # 2–4
        self.include_asm   = include_asm
        self.start_date    = pd.Timestamp(start_date)
        self.end_date      = pd.Timestamp(end_date)
        self.location_id   = location_id

    # ------------------------------------------------------------------
    # Build roster
    # ------------------------------------------------------------------
    def _build_employees(self) -> pd.DataFrame:
        used_pids  = set()
        used_names = set()
        employees  = []

        def _add(role: str, employment_type: str,
                 desired_min: int, desired_max: int,
                 pref_override: str = None):
            pref = pref_override or random.choice(PREFERENCE_LABELS)
            desired_hours = random.randint(desired_min, desired_max)

            if pref == "morning":
                avail_start, avail_end = 5, 15
            elif pref == "evening":
                avail_start, avail_end = 12, 22
            elif pref == "midday":
                avail_start, avail_end = 8, 18
            else:
                avail_start, avail_end = 5, 22

            unavail_days = random.sample(
                range(7),
                k=random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
            )

            # Managers and supervisors are less likely to call out
            if role in ("Store Manager", "Assistant Store Manager"):
                no_show_risk = round(rng.uniform(0.00, 0.04), 3)
                pay_rate     = round(rng.uniform(50_000, 75_000) / 2080, 2)  # salaried equivalent
                tenure       = random.randint(12, 120)
            elif role == "Shift Supervisor":
                no_show_risk = round(rng.uniform(0.01, 0.08), 3)
                pay_rate     = round(rng.uniform(20.0, 28.0), 2)
                tenure       = random.randint(6, 84)
            else:
                no_show_risk = round(rng.uniform(0.02, 0.20), 3)
                pay_rate     = round(rng.uniform(15.0, 22.0), 2)
                tenure       = random.randint(1, 72)

            employees.append({
                "partner_id":            _unique_partner_id(used_pids),
                "name":                  _random_name(used_names),
                "location_id":           self.location_id,
                "role":                  role,
                # All partners are cross-trained on every station
                "skills":                ",".join(BARISTA_SKILLS),
                "employment_type":       employment_type,
                "desired_weekly_hours":  desired_hours,
                "preference":            pref,
                "avail_start_hour":      avail_start,
                "avail_end_hour":        avail_end,
                "unavailable_days":      ",".join(map(str, unavail_days)),
                "productivity_score":    round(rng.uniform(0.6, 1.0), 3),
                "no_show_risk":          no_show_risk,
                "fatigue_sensitivity":   random.choices([0, 1], weights=[0.70, 0.30])[0],
                "pay_rate":              pay_rate,
                "tenure_months":         tenure,
                "is_probationary":       0,   # resolved below
            })

        # ── Store Manager (1, salaried, flexible availability) ──────────
        _add("Store Manager", "full_time", 40, 50, pref_override="morning")

        # ── Assistant Store Manager (0 or 1) ─────────────────────────────
        if self.include_asm:
            _add("Assistant Store Manager", "full_time", 38, 48,
                 pref_override=random.choice(["morning", "flexible"]))

        # ── Shift Supervisors (2–4, always full-time) ────────────────────
        for _ in range(self.n_supervisors):
            _add("Shift Supervisor", "full_time", 32, 40)

        # ── Baristas (mix of full-time and part-time) ────────────────────
        for _ in range(self.n_baristas):
            emp_type = random.choices(
                ["full_time", "part_time"], weights=[0.30, 0.70]
            )[0]
            dmin, dmax = (32, 40) if emp_type == "full_time" else (12, 28)
            _add("Barista", emp_type, dmin, dmax)

        df = pd.DataFrame(employees)
        df["is_probationary"] = (df["tenure_months"] < 3).astype(int)
        return df

    # ------------------------------------------------------------------
    # Build weekly schedules
    # ------------------------------------------------------------------
    def _generate_schedules(self, employees: pd.DataFrame) -> pd.DataFrame:
        """
        Produce shift history rows. Schema intentionally omits 'role' to
        match real scheduling-system CSV exports.
        """
        records    = []
        week_start = self.start_date
        end        = self.end_date

        while week_start <= end:
            for _, emp in employees.iterrows():
                # Store/Assistant Managers appear on schedule ~60% of weeks
                if emp["role"] in ("Store Manager", "Assistant Store Manager"):
                    if rng.random() > 0.60:
                        continue
                    shift_pool = self.MANAGER_SHIFT_TYPES
                else:
                    shift_pool = self.SHIFT_TYPES

                unavail = set(
                    int(d) for d in emp["unavailable_days"].split(",") if d
                )
                weekly_hours = 0

                for day_offset in range(7):
                    shift_date = (week_start + timedelta(days=day_offset)).date()
                    if shift_date > end.date():
                        break
                    dow = shift_date.weekday()
                    if dow in unavail:
                        continue
                    if weekly_hours >= emp["desired_weekly_hours"]:
                        break

                    # Filter shift types to those inside the employee's
                    # availability window
                    compatible = {
                        k: v for k, v in shift_pool.items()
                        if (v[0] >= emp["avail_start_hour"] and
                            v[1] <= emp["avail_end_hour"])
                    }
                    if not compatible:
                        compatible = shift_pool

                    # Preference-weighted shift selection
                    pref    = emp["preference"]
                    weights = []
                    for stype, (s_hr, _) in compatible.items():
                        if pref == "morning":
                            w = 4 if s_hr <= 7 else 1
                        elif pref == "evening":
                            w = 4 if s_hr >= 13 else 1
                        elif pref == "midday":
                            w = 4 if 8 <= s_hr <= 11 else 1
                        else:
                            w = 1
                        weights.append(w)

                    chosen_type    = random.choices(list(compatible.keys()), weights=weights)[0]
                    s, e           = compatible[chosen_type]
                    shift_hours    = e - s
                    no_show        = int(rng.random() < emp["no_show_risk"])
                    manager_override = int(rng.random() < 0.08)
                    weekly_hours  += shift_hours

                    records.append({
                        "partner_id":                 emp["partner_id"],
                        "name":                       emp["name"],
                        "location_id":                self.location_id,
                        "date":                       shift_date,
                        "day_of_week":                dow,
                        "shift_type":                 chosen_type,
                        "shift_start_hour":           s,
                        "shift_end_hour":             e,
                        "scheduled_hours":            shift_hours,
                        "no_show":                    no_show,
                        "manager_override":           manager_override,
                        "week_start":                 week_start.date(),
                        "weekly_hours_accumulated":   weekly_hours,
                    })
            week_start += timedelta(days=7)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def generate(self):
        employees = self._build_employees()
        schedules = self._generate_schedules(employees)
        return employees, schedules


# ===========================================================================
# 3. CONVENIENCE RUNNER
# ===========================================================================

def generate_all(output_dir: str = ".") -> dict:
    """
    Generate all synthetic datasets and save to CSV.
    Returns a dict of DataFrames for in-memory use.

    Roster defaults:
      1 Store Manager + 1 ASM + 3 Shift Supervisors + 20 Baristas = 25 partners
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating demand data ...")
    demand_gen = DemandDataGenerator()
    demand_df  = demand_gen.generate()
    demand_path = os.path.join(output_dir, "demand_history.csv")
    demand_df.to_csv(demand_path, index=False)
    print(f"  Saved {len(demand_df):,} rows → {demand_path}")

    print("Generating partner roster and schedule history ...")
    sched_gen = EmployeeScheduleGenerator(
        n_baristas=20,
        n_supervisors=3,
        include_asm=True,
    )
    employees_df, schedules_df = sched_gen.generate()

    emp_path   = os.path.join(output_dir, "employees.csv")
    sched_path = os.path.join(output_dir, "schedule_history.csv")
    employees_df.to_csv(emp_path, index=False)
    schedules_df.to_csv(sched_path, index=False)

    # Print roster summary
    role_counts = employees_df["role"].value_counts()
    print(f"  Roster ({len(employees_df)} partners):")
    for role, count in role_counts.items():
        print(f"    {role}: {count}")
    print(f"  Saved → {emp_path}")
    print(f"  Saved {len(schedules_df):,} shift records → {sched_path}")

    return {
        "demand":    demand_df,
        "employees": employees_df,
        "schedules": schedules_df,
    }


if __name__ == "__main__":
    generate_all(output_dir="data")
