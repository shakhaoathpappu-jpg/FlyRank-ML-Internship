Capstone Report —

* Author: Shakhaoath Hossain Pappu
* Lane: Refresh / Content Opportunity Scoring
* Repo: https://github.com/shakhaoathpappu-jpg/FlyRank-ML-Internship/tree/main
* Date: 2026-09-24

Copy this file to `work/capstone_report.md` and fill it in as you build. Sections 1–8 mirror the Pass / Needs-Work rubric axes, so nothing here is optional. Sections 0 and 9 are paper sections: your deployed research paper must carry both, and they're here so you never rebuild them from memory at ship time.

## 0. Abstract

FlyRank clients publish thousands of pages that quietly lose search traffic long before anyone notices, and content teams need a defensible way to decide which ones deserve attention first. Using 18 months of the FlyRank ML Internship warehouse (`fact_content_daily_performance` and `dim_content`, 2025-01 through the sealed final month 2026-06), this project asks whether a page's staleness (time since last content update) and CTR gap (how far its click-through rate falls short of similarly-ranked pages) predict which currently-visible pages will lose clicks the following month. A page-month panel was built, a leakage-checked label was defined (next-month click decline), and a hand-written rule-based baseline was compared against a Gradient Boosting model on a time-ordered, never-touched sealed test split. Restricted to pages that already receive at least one click at decision time, the model reaches AUC 0.594 on the sealed test month versus 0.497 for the baseline — a modest but real lift — after an earlier, unscoped version of the same pipeline was found to produce a misleading AUC of 0.91–0.93 driven entirely by a zero-click labeling artifact that was detected and removed. The output is a ranked, reason-coded action queue (`refresh_now` / `monitor` / `no_action`) that a content-ops team can use to decide, each month, which pages to prioritize for a refresh with a limited budget.

## 1. Problem framing

**Decision supported:** given a fixed content-ops budget, which published pages should be prioritized for a refresh this cycle, and why.

**Unit of analysis:** one page (`content_hash_id`), for one calendar month (page-month grain), scoped within one client (`client_hash_id`).

**Output:** a ranked queue per page, with a continuous `model_score`, a categorical `action_label` (`refresh_now` / `monitor` / `no_action`), and a `reason_code` (`STALE`, `CTR_GAP`, `STALE_AND_CTR_GAP`, `NO_SIGNAL`) explaining why.

**Action a human takes:** a content editor pulls the `refresh_now` queue at the start of each cycle and works through it in rank order, using the reason code to decide what kind of refresh is needed (update stale content vs. fix metadata/CTR).

**Cost of a wrong call:** a false `refresh_now` wastes editor time on a page that didn't need it; a missed decline (false `no_action`) lets a page keep losing clicks for a full month before it's caught. Neither error is catastrophic, but both have a real opportunity cost against a limited editorial budget.

**Why ML helps here:** with 309,234 unique pages and 67 clients in this release, manual review of every page every month is infeasible. A ranked score turns an unbounded review problem into a fixed-size, prioritized queue.

## 2. Data safety

**Data used:** `fact_content_daily_performance` (daily Search Console metrics, aggregated here to page-month) and `dim_content` (content metadata — creation date, last-updated date, publish/deletion status), from the FlyRank ML Internship warehouse on Hugging Face.

**Deliberately excluded:**
- Any raw client name, domain, URL, or search query — the source data is already hashed (`client_hash_id`, `content_hash_id`), and no de-hashing or joining to external identity data was attempted anywhere in this project.
- GA4 and AI-referral columns (`ga4_*`, `ai_chatgpt`, `ai_perplexity`, etc.) — left out of the feature set on purpose, since this lane's hypothesis is specifically about Search Console signals (staleness, CTR, position), and AI-referral sessions are known to be too sparse for reliable use in this kind of model.
- Any column describing a month after the decision month — this is the single most important exclusion, enforced by construction in `build_pairs()`/`add_features()`, so the model never sees the future it's predicting.

**Leakage risks considered:**
- **Label-derived fields:** `trend_direction` and `future_clicks` are used only to *construct* the label (`needs_refresh`) and to verify signals during the Week-4 baseline (ML-07) — they are never passed into the model as a feature. This was checked explicitly with an assertion in the notebook.
- **The zero-click artifact:** the biggest leakage risk found in this project was not an obvious future-window leak, but a mechanical one — `needs_refresh` (clicks fall below the prior month) can never be true for a page with zero clicks, since clicks cannot go negative. Including those pages inflated an early version's AUC to 0.91–0.93. This was caught by comparing feature importances (near-total weight on `ctr`) against the hypothesis, then confirmed by re-running the evaluation restricted to pages with ≥1 click at decision time, where AUC dropped to an honest 0.59–0.61.
- **Pseudonymous IDs:** `client_hash_id` and `content_hash_id` are used only for grouping, merging monthly tables, and reporting rank order — never as model features.

**Confirmation:** no client name, domain, URL, or raw search query appears anywhere in `work/` — all identifiers in notebooks, outputs, and this report are the pre-hashed IDs supplied by the warehouse.

## 3. Baseline

**The rule (from Week-4 / ML-07):**

```
action_score = 0.5 × staleness_score + 0.5 × ctr_gap
```

where `staleness_score` is days since `content_updated_date`, capped at 180 days and scaled to [0, 1], and `ctr_gap` is how far a page's CTR falls below the median CTR of other pages in the same position bucket (1–3, 4–10, 11–20, 21+), clipped to [0, 1].

**Why it's a fair comparison:** it uses exactly the same two signals the model has access to (staleness and CTR-vs-position), computed from exactly the same decision-month data, and is evaluated on the identical time-aware split — so any gap between it and the model reflects the value of learning feature interactions, not access to different or better data.

**Its numbers, same data and metric as the model (AUC), scoped to pages with ≥1 click at decision time:**

| Split | Baseline AUC | n | Positive rate (base rate) |
|---|---|---|---|
| Validation | 0.511 | 322,953 | 52.9% |
| Sealed test | 0.497 | 76,295 | 62.6% |

The baseline's AUC sits at or below 0.50 on both splits — essentially no better than random ranking — which is itself an honest, useful finding: the two hand-picked signals alone don't separate decliners from non-decliners on their own.

## 4. Model / analysis

**Method:** `sklearn.ensemble.GradientBoostingClassifier` (`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `random_state=42`). Chosen because the lane calls for a ranked score with reason codes, not a black-box prediction, and gradient boosting on a handful of tabular features gives both a usable probability score and inspectable feature importances — consistent with keeping the model auditable, the way the rule-based baseline is.

**Feature list (all knowable at the decision month, none from the future):**
- `gsc_clicks`, `gsc_impressions`, `ctr`, `avg_position` — raw search performance in month *M*.
- `staleness_score` — days since last content update, capped/scaled.
- `ctr_gap` — CTR shortfall vs. the page's position-bucket median.

**Left out on purpose:** GA4/session-source columns, AI-referral columns, and raw `content_type` (not one-hot encoded, to keep the feature set small, interpretable, and directly comparable to the baseline's two inputs).

**Target / proxy definition (one sentence):** `needs_refresh = 1` if a page's clicks in the month after the decision month are lower than its clicks in the decision month itself, else `0`, restricted to pages with at least one click at decision time.

## 5. Evaluation

**Split:** time-aware, not random and not shuffled — training uses decision months 2025-01 through 2025-09, validation uses 2025-10 through 2026-04, and the sealed test uses a single pairing (decision month 2026-05 → outcome month 2026-06, the warehouse's real final month), touched only once. Chronological ordering was asserted in code (`assert max(train) < min(val) < min(test)`) to rule out any time overlap between splits.

**Metrics, model vs. baseline, same split:**

| Split | Approach | AUC | n | Positive rate (base rate) |
|---|---|---|---|---|
| Validation | Baseline (rule) | 0.511 | 322,953 | 52.9% |
| Validation | Model (GBM) | 0.611 | 322,953 | 52.9% |
| Sealed test | Baseline (rule) | 0.497 | 76,295 | 62.6% |
| Sealed test | Model (GBM) | **0.594** | 76,295 | 62.6% |

AUC is reported instead of accuracy specifically because the base (positive) rate is high (52.9%–62.6%) — a model that always predicted "will decline" would already score 62.6% accuracy on the sealed test with zero discrimination ability, so accuracy alone would be misleading here.

**What the errors look like:** the model's biggest blind spot is very-low-traffic pages — single-digit clicks and under-60 impressions — where month-to-month click counts are noisy enough that a real ranking-worthy decline and ordinary fluctuation look similar in the available features. This shows up as `refresh_now` picks with `avg_position` in the 50–95 range but only 1–2 total clicks, where the model's confidence is high but the absolute numbers are small enough that a single random visit could flip the label next month.

## 6. Interpretation

**Feature importances (Gradient Boosting):**

| Feature | Importance |
|---|---|
| `avg_position` | 0.561 |
| `ctr` | 0.197 |
| `gsc_impressions` | 0.166 |
| `gsc_clicks` | 0.072 |
| `ctr_gap` | 0.003 |
| `staleness_score` | 0.001 |

**In plain words:** a page's average search ranking position is by far the strongest predictor of whether it will lose clicks next month, followed by its current CTR and impression volume. `ctr_gap` and `staleness_score` — the two signals this whole lane's baseline was built around — contribute almost nothing once the label is leakage-corrected.

**Surprises and negative results:**
- The founding hypothesis (staleness drives refresh need) is not supported by this data — a well-understood negative result, not a failed project. `staleness_score` importance of 0.001 is about as close to "no effect" as a model produces.
- The single most important methodological finding was not a modeling result at all: the discovery that an early, unscoped version of this pipeline scored AUC 0.91–0.93 purely because zero-click pages can never register a "decline" by construction. Catching and correcting this — rather than reporting the flashier 0.91 number — is treated here as the project's most important result.

## 7. Recommendation

**Ranked action playbook (sealed test month, 76,295 eligible pages with ≥1 click):**

| Action | Pages | Share |
|---|---|---|
| `refresh_now` | 4,503 | 5.9% |
| `monitor` | 59,561 | 78.1% |
| `no_action` | 12,231 | 16.0% |

**How a FlyRank editor would use this tomorrow:** pull the `refresh_now` queue (ranked by `model_score`, highest first), and work top-down. Pages with reason code `STALE` (the large majority of `refresh_now`) typically combine low CTR (3–15%) with very poor ranking (average position 50–95, i.e. page 5+ of results) — these are reasonable first candidates for a content update or on-page fix, subject to the editor's own judgment about whether the page is meant to be actively maintained.

**Confidence and limits, stated explicitly:** the model's edge over the rule-based baseline is real but modest (AUC 0.59–0.61 vs. 0.50–0.51) — this is a useful way to *sort* a review queue, not a confident, page-by-page guarantee. It excludes zero-click pages entirely (a separate visibility/indexing problem, out of scope here), reflects one data release and one sealed test month, and makes no claim about *why* a refresh would help — only that these pages show the same signal pattern as pages that lost clicks historically.

## 8. Reproducibility

**Environment:** Google Colab (standard runtime), Python 3.13, with `huggingface_hub`, `pandas`, `numpy`, `scikit-learn`, and `matplotlib` (all pre-installed in Colab or installed via `pip install` in the notebook's setup cell).

**Random seed:** `random_state=42`, set on both `train_test_split` calls (ML-07) and `GradientBoostingClassifier` (capstone).

**Exact steps to re-run from a fresh clone:**
1. Open `work/notebooks/capstone_refresh_scoring.ipynb` in Google Colab.
2. Request access to `FlyRank/internship-warehouse` on Hugging Face, create a **Read** token, and add it as a Colab secret named `HF_TOKEN` (enable notebook access for this specific notebook).
3. Runtime → Run all. The notebook downloads and aggregates all 18 months of `fact_content_daily_performance` plus `dim_content`, builds the time-aware splits, trains the baseline and model, and regenerates `work/outputs/capstone_ranked_recommendations.csv` and the four figures under `work/outputs/figures/`.
4. No manual steps beyond providing `HF_TOKEN` are required.

**Sealed evaluation, checkable from the repo:**
- The cell that builds the sealed frame is the `build_pairs([TEST_DECISION_MONTH])` call in the Methodology section, where `TEST_DECISION_MONTH = "2026-05"` and its outcome month is the warehouse's real final month, `2026-06` — used exactly once for the numbers reported in Section 5.
- The metrics that call produced are committed as `work/outputs/capstone_metrics.json` (generated by the snippet below, added to the end of the Results section and re-run once before final commit):

```python
import json

metrics = {
    "validation": {
        "baseline_auc": float(baseline_val_auc),
        "model_auc": float(model_val_auc),
        "n": int(len(y_val)),
        "positive_rate": float(y_val.mean()),
    },
    "sealed_test": {
        "baseline_auc": float(baseline_test_auc),
        "model_auc": float(model_test_auc),
        "n": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
    },
    "feature_importance": feature_importance.to_dict(),
    "random_state": 42,
}

with open("work/outputs/capstone_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
```

This JSON file — unlike the CSV recommendation queue — **is** committed to git, so the "evaluated once, blind" claim in this report is checkable against the repo rather than taken on faith.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

**Claims checklist confirmed before submitting:**
- [x] Language throughout uses observed / measured / directional / decision-support framing — no causal claims are made anywhere in this report.
- [x] Base rates (52.9% validation, 62.6% sealed test) are reported alongside AUC, and AUC — not accuracy — is used as the headline metric specifically because the base rate is high enough that accuracy alone would be misleading.
- [x] No claim is made about predicting or reverse-engineering Google's ranking algorithm.
- [x] No client-identifying details (names, domains, URLs, raw queries) appear anywhere in this report or in `work/`.
- [x] The numbers in this report match a fresh top-to-bottom re-run of `work/notebooks/capstone_refresh_scoring.ipynb`.
