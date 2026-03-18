# P2-ETF Momentum Maxima — SDHO Enhanced

A systematic ETF momentum rotation strategy with overlays derived from the
physics of market dynamics. Selects one ETF (or cash) per session using a
multi-factor ranking engine, now enhanced with three signal improvements
grounded in the **Stochastic Damped Harmonic Oscillator (SDHO)** framework
published in Dean (2026).

**Data Infrastructure:** Historical prices stored in [Hugging Face Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-momentum-maxima) with daily incremental updates via GitHub Actions.

---

## What It Does

The strategy ranks a universe of ETFs daily across four factors, picks the
top-ranked asset, and holds it until the ranking changes or a risk gate
triggers a move to cash. It runs as a Streamlit web app with live data from
Yahoo Finance and FRED.

Two universes are available:

| Universe | ETFs |
|---|---|
| Option A — Fixed Income / Alts | GLD · SLV · VNQ · TLT · LQD · HYG · VCIT |
| Option B — Equities | SPY · QQQ · XLV · XLF · XLE · XLI · XLK · XLY · XLP · XLB · XLRE · XLU · XLC · XBI · XME · XHB · XSD · XRT · XAR · XNT · XNTK |

Benchmarks: SPY (equity), AGG (bond)

---

## How Signals Are Generated

Each day, every ETF in the active universe is scored across four factors.
The asset with the highest combined score is selected, provided its return
over the training window exceeds the rolling T-bill yield (the cash hurdle).
If no asset clears the hurdle, the strategy holds cash.

### Base ranking factors (max score 20)

| Factor | Computation | Max rank |
|---|---|---|
| Return rank | Total price return over the training window | 5 |
| Z-score rank | Cross-sectional z-score of returns vs universe mean | 5 |
| Acceleration rank | Fast return (⅓ of window) minus slow return | 5 |
| Volume confirmation | Rank 1–5 only if current volume exceeds prior window average; otherwise 0 | 5 |

The fast return minus slow return term (acceleration) is the
momentum-of-momentum signal: positive means a trend is strengthening,
negative means it is fading.

---

## SDHO Theory Background

> Dean, B.H. (2026). *Scale Invariant Dynamics in Market Price Momentum.*
> SSRN Working Paper.

The paper constructs a kinematic phase space for futures markets by plotting
price momentum (x) against its rate of change, or acceleration (y = Δx).
Applying sparse identification of nonlinear dynamics (SINDy) to 15 years of
E-mini S&P 500 data, it discovers that momentum follows a **Stochastic
Damped Harmonic Oscillator**:

```
dx/dt = y
dy/dt = -kx - Ωy + σ·dW
```

where x is momentum, y is acceleration, k is the restoring force, Ω is the
damping rate, and σ·dW is stochastic noise.

### Three universal parameters

| Parameter | Value | Description |
|---|---|---|
| Ω (dissipation rate) | ≈ 1.15 ± 0.04 | How fast momentum shocks decay. Universal across asset classes (CV < 4%). |
| R² (deterministic fraction) | ≈ 0.57 ± 0.01 | 57% of momentum acceleration is predictable from phase-space position; 43% is noise. Universal. |
| Φ (scaled restoring force) | Varies by market | k · Var(x) = σ²/(2Ω). Encodes mean-reversion intensity. Asset-specific. |

Φ ranges from 0.016 (currencies, weak mean-reversion) to 0.292 (energy,
strong mean-reversion), an 18× spread.

### Phase space quadrants

Because the SDHO is a two-dimensional dynamical system, every asset at every
moment occupies one of four quadrants defined by the sign of momentum (x)
and acceleration (y):

```
         Acceleration y
              (+)
               |
    Q4         |         Q1
 Crash         |      Accelerating
 recovery      |         rally
               |
  ─────────────+───────────── Momentum x
               |
    Q3         |         Q2
 Accelerating  |      Rally
 decline       |      fading ⚠️
               |
              (-)
```

The SDHO flow field predicts that assets in **Q2 (positive momentum,
negative acceleration)** are being pulled back toward equilibrium by the
restoring force −kx. Mean reversion is imminent. Without intervention, a
pure return-rank system will continue to favour these assets because their
historical return is still high, even as the signal is degrading.

---

## SDHO Enhancements

Three enhancements are implemented as toggleable sidebar controls. All three
are on by default.

### 1 · Φ-calibrated look-back windows

**The problem:** A uniform look-back window treats a Treasury ETF the same
as an energy ETF. But energy (high Φ) mean-reverts much faster than
Treasuries (low Φ). Using the same window for both leaves signal on the table.

**The fix:** Each ETF's look-back is multiplied by a Φ-derived scalar.

| ETF | Asset class | Φ analog | Multiplier | Effect |
|---|---|---|---|---|
| XLE | Energy | CL futures Φ=0.292 | 0.60× | Shorter window — catches reversals earlier |
| GLD / SLV | Precious metals | GC futures Φ=0.106 | 0.75× | Moderately shorter |
| VNQ | Real estate | Between equity and bonds | 0.90× | Slightly shorter |
| SPY / QQQ / XLV / XLF / XLI | Equities | ES futures Φ=0.046–0.069 | 1.00× | Baseline |
| HYG | High yield bonds | Credit-sensitive | 1.10× | Slightly longer |
| VCIT | Intermediate corp bonds | Below equity | 1.25× | Longer |
| LQD | Long corp bonds | ZB futures Φ=0.021 | 1.30× | Longer |
| TLT | Long Treasuries | Near ZB/currency | 1.40× | Longest window — trends persist |

The fast acceleration window also scales with each asset's calibrated
look-back, keeping the 1:3 fast-to-slow ratio intact.

**Where it lives in the code:**
```python
# ingestor.py and streamlit_app.py
PHI_LOOKBACK_MULTIPLIER = { 'XLE': 0.60, 'TLT': 1.40, ... }

def get_asset_training_days(ticker: str, base_days: int) -> int:
    mult = PHI_LOOKBACK_MULTIPLIER.get(ticker, 1.0)
    return max(10, int(base_days * mult))
```

---

### 2 · Phase quadrant filter

**The problem:** An asset with strong historical returns but decelerating
momentum (Q2: x>0, y<0) still scores highly on return rank and z-score rank.
The system selects it just as the SDHO flow field predicts mean reversion.

**The fix:** Apply a rank adjustment based on phase-space quadrant.

```python
rally_fading   = (rets > 0) & (accel < 0)   # Q2: imminent mean reversion
crash_recovery = (rets < 0) & (accel > 0)   # Q4: potential reversal

rank_sum += where(rally_fading,   penalty,  0)   # default: -3
rank_sum += where(crash_recovery, bonus,    0)   # default: +1
```

The penalty is tunable in the sidebar (range −1 to −6). A value of −3 means
a Q2 asset needs to beat the next-best asset by 3 rank points on the other
factors to still be selected. The crash recovery bonus is intentionally small
(+1) to avoid over-weighting distressed assets.

The ranking matrix in the UI now shows each asset's current quadrant label
so the effect is fully auditable.

**Backtest guidance:** Start with penalty = −3, test across −1 to −5. Larger
penalties are more conservative; they will reduce whipsaw trades into fading
momentum but may delay entry on renewed trends. A penalty of −6 effectively
blocks all Q2 assets.

---

### 3 · Ω re-entry cooldown

**The problem:** After a stop-loss trigger, the strategy can whipsaw back
into a position on the same day or the next day if the z-score recovers
quickly. This incurs transaction costs and re-enters into residual momentum
decay.

**The fix:** The SDHO places the momentum half-life at:

```
t½ = ln(2) / |λ₁|
```

where λ₁ ≈ −0.28 is the slow eigenmode of the system matrix. At hourly
aggregation this gives t½ ≈ 2.5 hours. At daily resolution this translates
to approximately 1 trading day.

After a stop trigger, the z-score exit condition is not evaluated until at
least `MIN_CASH_BARS_AFTER_STOP = 1` full bars have elapsed in cash. This
prevents re-entry during the initial momentum decay phase without
meaningfully delaying recovery.

```python
if stop_active:
    cash_bars_held += 1
    cooldown_expired = (cash_bars_held >= min_cash_bars)
    if cooldown_expired and mz <= z_exit_threshold:
        stop_active = False
```

---

## Risk Gates

These are independent of the SDHO enhancements and remain as original.

| Gate | Description |
|---|---|
| Cash hurdle | Asset return over training window must exceed rolling T-bill yield |
| Trailing stop | If 2-day cumulative return ≤ threshold (default −12%), switch to cash |
| Z-score re-entry | Exit cash when max cross-sectional z-score recovers above threshold (default 0.9) |
| Volatility filter (optional) | Exclude assets with 20-day annualised vol above threshold |
| 200-day MA filter (optional) | Exclude assets trading below their 200-day simple moving average |

---

## Data Architecture

The system uses **Hugging Face Datasets** as the canonical data store,
replacing the previous GitLab file storage architecture.

### Data Flow

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Yahoo Finance  │      │  GitHub Actions  │      │  Hugging Face   │
│  + FRED (TBill) │─────▶│  (22:30 UTC)     │─────▶│  Dataset        │
│  Daily Prices   │      │  Incremental     │      │  P2SAMAPA/      │
└─────────────────┘      │  Update Script   │      │  p2-etf-mom...  │
                         └──────────────────┘      └────────┬────────┘
                                                            │
                         ┌──────────────────┐               │
                         │  Streamlit App   │◄──────────────┘
                         │  (hf_hub_download)│     Public Read
                         └──────────────────┘
```

### Key Features

- **Incremental Updates:** Only new daily bars are fetched and appended (vs. rebuilding 18 years of history)
- **Public Access:** Dataset is public; Streamlit requires no API keys to read
- **Version Control:** HF Dataset commit history tracks every daily update
- **Format:** Parquet with MultiIndex columns `(Ticker, Close/Volume)` + `(CASH, Daily_Rf)`

---

## File Structure

```
├── streamlit_app.py          # Main app: signal generation, backtest, UI
│                             # Reads from HF Dataset via hf_hub_download
│
├── ingestor_incremental.py   # Daily incremental data fetcher
│                             # Appends new bars to existing HF Dataset
│
├── requirements.txt          # Python dependencies (includes huggingface-hub)
│
├── scripts/
│   └── seed_hf_dataset.py    # One-time full history rebuild (18+ years)
│                             # Run manually to initialize HF Dataset
│
└── .github/workflows/
    ├── seed_hf_dataset.yml   # One-time seeding workflow (manual trigger)
    └── sync_to_hf.yml        # Daily incremental sync (22:30 UTC cron)
```

---

## Setup

### Prerequisites

- Python 3.10+
- Hugging Face account (for dataset write access; read is public)
- FRED API access (public, no key required for DTB3 CSV endpoint)

### Installation

```bash
git clone <your-repo-url>
cd P2-ETF-MOMENTUM-MAXIMA
pip install -r requirements.txt
```

### Configuration

**For Local Development:**
No configuration required for reading data (public dataset).

**For GitHub Actions (Data Updates):**
Add your Hugging Face token to GitHub Secrets:
1. Generate token at https://huggingface.co/settings/tokens (Write permission)
2. Add to GitHub → Settings → Secrets and variables → Actions:
   - Name: `HF_TOKEN`
   - Value: `hf_...`

**For Streamlit Cloud Deployment:**
No secrets required for the dataset (public read). If you later make the dataset private, add:
```toml
# .streamlit/secrets.toml
HF_TOKEN = "hf_..."
```

### Initialize the Dataset (One-time)

Run the seeding workflow to populate HF Dataset with full history:

```bash
# Via GitHub Actions (recommended):
# Actions → Seed HF Dataset (Full Rebuild) → Run workflow

# Or locally (if you have HF_TOKEN set):
python scripts/seed_hf_dataset.py
```

### Running the App Locally

```bash
streamlit run streamlit_app.py
```

The app will automatically download the dataset from Hugging Face on first load (cached for 1 hour).

---

## Tuning the SDHO Parameters

All three SDHO enhancements expose tunable controls in the sidebar. The
table below gives guidance on when to adjust each parameter.

| Control | Default | Increase if... | Decrease if... |
|---|---|---|---|
| Rally fading penalty | −3 | Too many whipsaw trades into fading rallies | Missing strong renewed trends after a brief deceleration |
| Crash recovery bonus | +1 | Universe has many mean-reverting assets | Getting pulled into distressed positions too early |
| Φ multipliers | Per-table | (Edit code) asset trends faster than expected | (Edit code) asset mean-reverts faster than expected |
| Ω cooldown bars | 1 | Whipsawing back into stop situations | Missing re-entries after sharp, brief drawdowns |

---

## Limitations and Caveats

**The SDHO paper uses futures; this system trades ETFs.** The Φ values are
mapped from nearest futures analogs (CL → XLE, ZB → TLT, etc.). ETF
microstructure differs from futures — creation/redemption arbitrage, tracking
error, and liquidity differences mean the exact Φ values may not transfer
directly. The mappings are economically motivated approximations.

**Data Latency:** The daily update runs at 22:30 UTC (5:30 PM EST) after market close. 
FRED T-bill data may lag by 1 business day; the system forward-fills the last available rate.

**All backtests are in-sample.** The paper's R² ≈ 0.57 is itself an
in-sample conditional variance decomposition of momentum acceleration, not
a return predictability claim. Out-of-sample performance will differ from
backtested results.

**Parameters should be treated as stable but not static.** The paper notes
that Ω and R² are stable over 15 years of futures data but acknowledges
they may shift in extreme regimes. A rolling-window analysis of how the
parameters evolve through VIX regimes would be informative.

**The linear SDHO applies at intraday-to-weekly timescales.** At monthly
and longer horizons the paper notes nonlinear terms become significant.
This strategy uses daily data with training windows of 3–18 months, which
sits near the upper boundary of where the linear model is reliable.

**This is not investment advice.** Past performance is not indicative of
future results. The strategy is a research tool.

---

## References

Dean, B.H. (2026). *Scale Invariant Dynamics in Market Price Momentum.*
SSRN Working Paper (v1.2.3, March 2026). ORCID: 0009-0008-8153-3269.

Brunton, S.L., Proctor, J.L., and Kutz, J.N. (2016). Discovering governing
equations from data by sparse identification of nonlinear dynamics. *PNAS*,
113(15), 3932–3937.

Lo, A.W. (2004). The adaptive markets hypothesis. *Journal of Portfolio
Management*, 30(5), 15–29.

Jegadeesh, N., and Titman, S. (1993). Returns to buying winners and selling
losers. *Journal of Finance*, 48(1), 65–91.
