"""Uber convertible notes — Tsiveriotis-Fernandes (1998) binomial lattice valuation engine.

Instruments: 0.875% Convertible Senior Notes due 2028 and 0% Convertible Senior Notes due 2025.
All values per $1,000 principal unless stated; rates continuously compounded.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------------------------------------------- instruments
@dataclass
class Note:
    name: str
    principal_mm: float        # outstanding principal, $M
    coupon: float              # annual coupon rate
    freq: int                  # coupons per year
    maturity: date
    first_coupon: date | None  # first coupon date
    conv_rate: float           # shares per $1,000
    call_start: date           # provisional (soft) call allowed on/after
    call_trigger: float = 1.30 # stock >= trigger x conversion price
    face: float = 1000.0

    @property
    def conv_price(self):
        return self.face / self.conv_rate

    def coupon_dates(self):
        if self.coupon == 0 or self.first_coupon is None:
            return []
        out, d = [], self.first_coupon
        while d <= self.maturity:
            out.append(d)
            m = d.month + 12 // self.freq
            d = date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, d.day)
        return out


NOTE_2028 = Note("2028 Notes (0.875%)", 1725.0, 0.00875, 2, date(2028, 12, 1), date(2024, 6, 1),
                 13.7848, date(2026, 12, 5))
NOTE_2025 = Note("2025 Notes (0%)", 1150.0, 0.0, 2, date(2025, 12, 15), None,
                 12.3701, date(2023, 12, 20))

# ---------------------------------------------------------------- market data
# Treasury par yields (%), US Treasury daily par yield curve. Tenors in years.
TENORS = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
CURVES = {
    date(2024, 3, 28): [5.49, 5.48, 5.46, 5.42, 5.38, 5.03, 4.59, 4.40, 4.21, 4.20, 4.20, 4.45, 4.34],
    date(2024, 6, 28): [5.47, 5.47, 5.48, 5.45, 5.33, 5.09, 4.71, 4.52, 4.33, 4.33, 4.36, 4.61, 4.51],
    date(2024, 9, 30): [4.93, 4.87, 4.73, 4.65, 4.38, 3.98, 3.66, 3.58, 3.58, 3.67, 3.81, 4.19, 4.14],
    date(2024, 12, 31): [4.40, 4.39, 4.37, 4.32, 4.24, 4.16, 4.25, 4.27, 4.38, 4.48, 4.58, 4.86, 4.78],
}
GS5_NOV2023 = 4.49   # FRED GS5, Nov-2023 monthly average (daily curve for 20-Nov-2023 not retrievable)

# ICE BofA option-adjusted spreads (%), FRED BAMLH0A1HYBB (BB) and BAMLC0A4CBBB (BBB)
OAS_BB = {date(2023, 11, 20): 2.51, date(2024, 3, 28): 1.87, date(2024, 6, 28): 1.84,
          date(2024, 9, 30): 1.89, date(2024, 12, 31): 1.86}
OAS_BBB = {date(2023, 11, 20): 1.46, date(2024, 3, 28): 1.15, date(2024, 6, 28): 1.17,
           date(2024, 9, 30): 1.16, date(2024, 12, 31): 1.02}
UPGRADE = date(2024, 8, 16)  # S&P BB+ -> BBB- (Fitch BBB 20-Aug, Moody's Baa2 27-Aug-2024)

# Uber month-end closes (digrin.com), used for spot and trailing realized volatility
MONTHLY_CLOSE = {
    "2019-05": 40.41, "2019-06": 46.38, "2019-07": 42.14, "2019-08": 32.57, "2019-09": 30.47, "2019-10": 31.50,
    "2019-11": 29.60, "2019-12": 29.74, "2020-01": 36.29, "2020-02": 33.87, "2020-03": 27.92, "2020-04": 30.27,
    "2020-05": 36.32, "2020-06": 31.08, "2020-07": 30.26, "2020-08": 33.63, "2020-09": 36.48, "2020-10": 33.41,
    "2020-11": 49.66, "2020-12": 51.00, "2021-01": 50.93, "2021-02": 51.75, "2021-03": 54.51, "2021-04": 54.77,
    "2021-05": 50.83, "2021-06": 50.12, "2021-07": 43.46, "2021-08": 39.14, "2021-09": 44.80, "2021-10": 43.82,
    "2021-11": 38.00, "2021-12": 41.93, "2022-01": 37.40, "2022-02": 36.03, "2022-03": 35.68, "2022-04": 31.48,
    "2022-05": 23.20, "2022-06": 20.46, "2022-07": 23.45, "2022-08": 28.76, "2022-09": 26.50, "2022-10": 26.57,
    "2022-11": 29.14, "2022-12": 24.73, "2023-01": 30.93, "2023-02": 33.26, "2023-03": 31.70, "2023-04": 31.05,
    "2023-05": 37.93, "2023-06": 43.17, "2023-07": 49.46, "2023-08": 47.23, "2023-09": 45.99, "2023-10": 43.28,
    "2023-11": 56.38, "2023-12": 61.57, "2024-01": 65.27, "2024-02": 79.50, "2024-03": 76.99, "2024-04": 66.27,
    "2024-05": 64.56, "2024-06": 72.68, "2024-07": 64.47, "2024-08": 73.13, "2024-09": 75.16, "2024-10": 72.05,
    "2024-11": 71.96, "2024-12": 60.32, "2025-01": 66.85, "2025-02": 76.01, "2025-03": 72.86, "2025-04": 81.01,
    "2025-05": 84.16, "2025-06": 93.30, "2025-07": 87.75, "2025-08": 93.75, "2025-09": 97.97, "2025-10": 96.50,
    "2025-11": 87.54, "2025-12": 81.71}

# Issue / capped-call data (pricing release 20-Nov-2023; 10-K FY2024)
ISSUE_DATE = date(2023, 11, 20)
ISSUE_SPOT = 54.75
CAP_PRICE = 95.81
CAPPED_CALL_COST_MM = 141.0      # for the full $1,725M (incl. option); $122.25M on the base $1,500M
NET_PROCEEDS_MM = 1700.0

# Disclosed fair values ($bn, Level 2, rounded to $0.1bn in the filings)
DISCLOSED = {  # date: (spot, {note: fv_bn})
    date(2024, 3, 28): (76.99, {"2028": 2.1, "2025": 1.3}),
    date(2024, 6, 28): (72.68, {"2028": 2.1, "2025": 1.2}),
    date(2024, 9, 30): (75.16, {"2028": 2.2, "2025": 1.3}),
    date(2024, 12, 31): (60.32, {"2028": 1.9, "2025": 1.2}),
}
NOTES = {"2028": NOTE_2028, "2025": NOTE_2025}


# ---------------------------------------------------------------- helpers
def yf(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.25


def cc(y_pct: float) -> float:
    """semi-annual bond-equivalent par yield (%) -> continuous rate"""
    return 2 * math.log(1 + y_pct / 200)


def rf(val_date: date, T: float) -> float:
    if val_date not in CURVES:
        return cc(GS5_NOV2023)
    return cc(float(np.interp(T, TENORS, CURVES[val_date])))


def spread(val_date: date, basis: str = "rating") -> float:
    """credit spread (decimal). 'rating': BB OAS before the S&P upgrade, BBB OAS after."""
    if basis == "BB":
        return OAS_BB[val_date] / 100
    if basis == "BBB":
        return OAS_BBB[val_date] / 100
    return (OAS_BB if val_date < UPGRADE else OAS_BBB)[val_date] / 100


def realized_vol(val_date: date, months: int = 36) -> float:
    keys = sorted(MONTHLY_CLOSE)
    end = f"{val_date.year}-{val_date.month:02d}"
    if val_date.day < 28:  # use last completed month
        m = val_date.month - 1 or 12
        y = val_date.year - (val_date.month == 1)
        end = f"{y}-{m:02d}"
    i = keys.index(end)
    px = np.array([MONTHLY_CLOSE[k] for k in keys[i - months:i + 1]])
    return float(np.std(np.diff(np.log(px)), ddof=1) * math.sqrt(12))


# ---------------------------------------------------------------- closed forms
def bs_call(S, K, T, r, sig, q=0.0):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sig * math.sqrt(T))


def capped_call_value(S, K, cap, T, r, sig):
    return bs_call(S, K, T, r, sig) - bs_call(S, cap, T, r, sig)


def capped_call_implied_vol(r=None, T=None):
    shares = NOTE_2028.principal_mm * 1e6 / 1000 * NOTE_2028.conv_rate
    target = CAPPED_CALL_COST_MM * 1e6 / shares
    T = T or yf(ISSUE_DATE, NOTE_2028.maturity)
    r = cc(GS5_NOV2023) if r is None else r
    return brentq(lambda s: capped_call_value(ISSUE_SPOT, NOTE_2028.conv_price, CAP_PRICE, T, r, s) - target, 0.05, 1.5), target


def straight_bond(note: Note, val_date: date, r: float, s: float) -> float:
    """PV of remaining coupons + principal at r + s (cash flows on actual dates)."""
    pv = 0.0
    for d in note.coupon_dates():
        if d > val_date:
            pv += note.face * note.coupon / note.freq * math.exp(-(r + s) * yf(val_date, d))
    return pv + note.face * math.exp(-(r + s) * yf(val_date, note.maturity))


def tf_european(note: Note, val_date: date, S, sig, r, s) -> float:
    """TF value with conversion only at maturity and no call: CR*S*N(d1) + F e^{-(r+s)T} N(-d2) + PV coupons."""
    T = yf(val_date, note.maturity)
    K = note.conv_price
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    cpn = straight_bond(note, val_date, r, s) - note.face * math.exp(-(r + s) * T)
    return note.conv_rate * S * norm.cdf(d1) + note.face * math.exp(-(r + s) * T) * norm.cdf(-d2) + cpn


# ---------------------------------------------------------------- TF lattice
def tf_lattice(note: Note, val_date: date, S0, sig, r, s, N=500, call=True, early_conv=True,
               return_grid=False):
    """Tsiveriotis-Fernandes CRR lattice.
    V = total convertible value, B = cash-only component (discounted at r+s); V-B discounted at r.
    Coupons are paid at the first step on/after each coupon date. Soft call modelled as forced
    conversion at any node on/after call_start where S >= trigger x conversion price."""
    T = yf(val_date, note.maturity)
    dt = T / N
    u = math.exp(sig * math.sqrt(dt)); d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    dr, drs = math.exp(-r * dt), math.exp(-(r + s) * dt)
    F, CR = note.face, note.conv_rate
    cpn_amt = F * note.coupon / note.freq
    cpn = np.zeros(N + 1)
    for cd in note.coupon_dates():
        if cd > val_date:
            k = min(N, math.ceil(yf(val_date, cd) / dt - 1e-9))
            cpn[k] += cpn_amt
    t_call = yf(val_date, note.call_start)
    trig = note.call_trigger * note.conv_price
    j = np.arange(N + 1)
    S = S0 * u ** (N - 2 * j)
    conv = CR * S
    V = np.maximum(conv, F) + cpn[N]
    B = np.where(F >= conv, F, 0.0) + cpn[N]
    for k in range(N - 1, -1, -1):
        S = S0 * u ** (k - 2 * np.arange(k + 1))
        conv = CR * S
        E = p * (V[:-1] - B[:-1]) + (1 - p) * (V[1:] - B[1:])
        Bc = drs * (p * B[:-1] + (1 - p) * B[1:])
        Vc = dr * E + Bc
        if call and k * dt >= t_call - 1e-12:
            forced = S >= trig
            Vc = np.where(forced, conv, Vc); Bc = np.where(forced, 0.0, Bc)
        if early_conv:
            cv = conv > Vc
            Vc = np.where(cv, conv, Vc); Bc = np.where(cv, 0.0, Bc)
        V, B = Vc + cpn[k], Bc + cpn[k]
    return (float(V[0]), float(B[0])) if not return_grid else (float(V[0]), float(B[0]), u, p)


def value(note_key: str, val_date: date, S0=None, sig=None, s_basis="rating", N=500, **kw):
    note = NOTES[note_key]
    T = yf(val_date, note.maturity)
    S0 = S0 if S0 is not None else DISCLOSED[val_date][0]
    r = rf(val_date, T); s = spread(val_date, s_basis)
    V, B = tf_lattice(note, val_date, S0, sig, r, s, N=N, **kw)
    return dict(V=V, B=B, r=r, s=s, T=T, S=S0, parity=note.conv_rate * S0,
                floor=straight_bond(note, val_date, r, s))


def implied_vol(note_key, val_date, target_per_1000, s_basis="rating", N=400):
    f = lambda v: value(note_key, val_date, sig=v, s_basis=s_basis, N=N)["V"] - target_per_1000
    lo, hi = 0.01, 1.5
    if f(lo) > 0:
        return float("nan")  # value above target even at ~zero vol
    return brentq(f, lo, hi)


def implied_spread_at_par(sig, price=1000.0, N=500):
    T = yf(ISSUE_DATE, NOTE_2028.maturity); r = cc(GS5_NOV2023)
    f = lambda s: tf_lattice(NOTE_2028, ISSUE_DATE, ISSUE_SPOT, sig, r, s, N=N)[0] - price
    return brentq(f, -0.02, 0.30)


# ---------------------------------------------------------------- accounting exhibit
def legacy_ccf_split(sig, s_issue):
    """Pre-ASU 2020-06 cash conversion model (ASC 470-20): liability = straight-debt FV, equity = residual."""
    r = cc(GS5_NOV2023)
    liab = straight_bond(NOTE_2028, ISSUE_DATE, r, s_issue) / 1000 * NOTE_2028.principal_mm
    eq = NOTE_2028.principal_mm - liab
    # effective rate that accretes the liability to par with the 0.875% coupon (semi-annual, ~5 years)
    n = 10; c = 1000 * NOTE_2028.coupon / 2; pv = liab / NOTE_2028.principal_mm * 1000
    y = brentq(lambda y: sum(c / (1 + y / 2) ** i for i in range(1, n + 1)) + 1000 / (1 + y / 2) ** n - pv, 0.0, 0.5)
    return liab, eq, y


if __name__ == "__main__":
    sig_cc, tgt = capped_call_implied_vol()
    print(f"capped call per share {tgt:.3f}, implied vol {sig_cc:.4f}")
    for dte, (S, fv) in DISCLOSED.items():
        for k in ("2028", "2025"):
            v = value(k, dte, sig=sig_cc)
            rv = realized_vol(dte)
            v2 = value(k, dte, sig=rv)
            print(dte, k, S, f"r={v['r']:.4f} s={v['s']:.4f} model={v['V']/10:.2f} rv={rv:.3f} model_rv={v2['V']/10:.2f}",
                  "reported", round(fv[k] * 1e3 / NOTES[k].principal_mm * 100, 2))
