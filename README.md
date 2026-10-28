# Uber Convertible Notes — Fair Value & ASC 815 Analysis (Tsiveriotis-Fernandes lattice)

Independent re-performance of the fair value of **Uber Technologies' 0.875% Convertible Senior Notes due 2028 ($1,725M)** and **0% Convertible Senior Notes due 2025 ($1,150M)**, built from public SEC filings and market data. It includes the ASC 815-40 / ASC 470-20 (ASU 2020-06) classification analysis.

## Headline: no inputs fitted to the answer
Volatility is Uber's trailing 36-month realized volatility, and the credit spread is the ICE BofA index for Uber's rating at each date. The model is then compared with the fair values Uber disclosed (Level 2, rounded to $0.1bn).

| Date | 2028 Notes: model | Reported | 2025 Notes: model | Reported |
|---|---|---|---|---|
| 28-Mar-2024 | $2.19bn | $2.1bn ✗ | $1.27bn | $1.3bn ✓ |
| 28-Jun-2024 | $2.11bn | $2.1bn ✓ | $1.24bn | $1.2bn ✓ |
| 30-Sep-2024 | $2.18bn | $2.2bn ✓ | $1.26bn | $1.3bn ✓ |
| 31-Dec-2024 | $1.91bn | $1.9bn ✓ | $1.17bn | $1.2bn ✓ |

**7 of 8 fall inside the ±$0.05bn rounding band**, with a mean absolute error of 2.1%. The miss follows a 22% rally in February 2024: the reported figure implies 35% volatility against 45% realized.

## Issue-date evidence (20-Nov-2023)
- **Capped calls:** Uber paid $5.93 per share for a $72.54/$95.81 call spread, more than any flat-vol Black-Scholes value (maximum $4.90). The price implies 46% volatility with about 3 points of skew. Flat-vol models understate call spreads and warrants of this kind.
- **New-issue discount:** the notes were sold at par against a model value of about 103.6. Par implies 39% volatility at the BB spread.

## Accounting
- The conversion option is indexed to Uber's stock and equity-classified (ASC 815-40), so it falls under the ASC 815-10-15-74(a) scope exception and is not bifurcated.
- After ASU 2020-06 the note is one liability at a 1.1% effective rate. The legacy cash-conversion model would have split it into a $1,278M liability (7.1% effective rate) and $447M of equity.
- Had bifurcation been required, Uber would have carried a $450–840M derivative, with quarterly earnings swings of up to about $260M.
- The capped calls are equity: recorded in APIC and not remeasured.

## Files
| File | What it is |
|---|---|
| `uber_convertible_model.xlsx` | Formula-driven Excel model with a live 100-step TF lattice for each of 9 cases (142,523 formulas, 0 errors). Matches Python to 4 decimals. Includes the issue-date capped-call and skew analysis, the ASC 815/470-20 analysis, sensitivities, issues log and sources |
| `uber_convertible_validation.ipynb` | Python validation: lattice convergence and closed-form check, issue-date evidence, out-of-sample quarter-end test, implied-vol backsolve, value profile and delta, sensitivities, accounting exhibit |
| `convtf.py` | Valuation engine (TF lattice, closed forms, capped call, implied vol and spread, legacy CCF split) |
| `uber_convertible_valuation_memo.pdf` | Client-style memo with a mock auditor Q&A |

## Method
Tsiveriotis-Fernandes (1998) splits the convertible into a cash-only component, discounted at r + s, and an equity component, discounted at r. The model uses a CRR tree with semi-annual coupons and models the soft call (130% trigger) as forced conversion. Holders may convert at any node; with no dividend, early conversion is never optimal. Risk-free rates come from the Treasury par curve interpolated to each note's term. Credit uses the BB OAS before S&P's 16-Aug-2024 upgrade and the BBB OAS after.

## Run
```bash
pip install numpy scipy pandas matplotlib jupyter
jupyter notebook uber_convertible_validation.ipynb
```

*Prepared by Parnika Sabharwal. Educational project based on public information; not a valuation opinion and not affiliated with Uber.*
