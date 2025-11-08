# Global Fishing Watch Vessel Enrichment Pipeline

**Date**: November 6, 2025  
**Purpose**: Enrich simulated vessel data with real-world behavioral events from Global Fishing Watch

---

## Overview

This pipeline creates **6 separate enriched datasets** that link your simulated vessels (by MMSI) to real-world data from the Global Fishing Watch API. Each dataset focuses on a specific aspect of vessel behavior and is designed to be used independently or combined for comprehensive analysis.

### What This Does

- Looks up your 332 simulated vessels in the Global Fishing Watch database
- Fetches historical behavioral events (encounters, gaps, loitering, port visits)
- Calculates risk assessments for potential illegal activity
- Creates separate CSV datasets for each event type, all linked by MMSI

### Data Coverage

- **Historical Range**: Typically 2012-2025 (varies by vessel)
- **Default Analysis Period**: 365 days back from present
- **Expected Match Rate**: 60-80% of simulated vessels found in GFW
- **Total Processing Time**: ~60 minutes for all 332 vessels

---

## Quick Start

### Step 1: Set API Token

```bash
export GFW_API_TOKEN='your_token_here'
```

Get token from: https://globalfishingwatch.org/our-apis/tokens

### Step 2: Mini Test (IMPORTANT!)

```bash
python3 test_enrichment_mini.py
```

This tests with 5 vessels (~2 minutes). **Check results before proceeding!**

### Step 3: Full Enrichment

```bash
python3 run_all_enrichment.py
```

Processes all 332 vessels (~60 minutes).

---

## Enriched Datasets

### 1. Vessel Identity (`vessel_identity_enrichment.csv`)
Links simulated vessels to real GFW identity (flag, IMO, AIS history)

### 2. Encounters/STS Transfers (`encounters_enrichment.csv`)
Ship-to-ship transfer events (potential transshipment/smuggling)

### 3. Dark Periods/Gaps (`gaps_enrichment.csv`)
AIS gap events (transponder turned off - evasion indicator)

### 4. Loitering Events (`loitering_enrichment.csv`)
Extended periods in one location (potential illegal activity)

### 5. Port Visits (`port_visits_enrichment.csv`)
Port visit history (where vessels docked)

### 6. Risk Assessment (`risk_assessment_enrichment.csv`)
Comprehensive risk scores (LOW/MEDIUM/HIGH) based on all events

**All datasets are linked by MMSI** to your simulation data.

---

## File Structure

```
demo-data-generation/
├── README.md                          # This file
├── enrich_vessel_identity.py          # Script 1: Identity lookup
├── enrich_encounters.py               # Script 2: Encounters
├── enrich_gaps.py                     # Script 3: Dark periods
├── enrich_loitering.py                # Script 4: Loitering
├── enrich_port_visits.py              # Script 5: Port visits
├── enrich_risk_assessment.py          # Script 6: Risk scores
├── test_enrichment_mini.py            # Mini test (5 vessels)
├── run_all_enrichment.py              # Master orchestrator
└── enrichment_outputs/                # Output CSVs
```

---

## API Rate Limiting

- **Delay**: 0.5s between requests
- **Retry**: 3 attempts on errors
- **Processing**: Sequential only

### Estimated Times

| Task | Vessels | Time |
|------|---------|------|
| Mini test | 5 | ~2 min |
| Identity | 332 | ~11 min |
| Each event type | 250 | ~10 min |
| **Total** | **332** | **~60 min** |

---

## Example Usage

```python
import pandas as pd

# Load datasets
identity = pd.read_csv('enrichment_outputs/vessel_identity_enrichment.csv')
risk = pd.read_csv('enrichment_outputs/risk_assessment_enrichment.csv')
encounters = pd.read_csv('enrichment_outputs/encounters_enrichment.csv')

# Find high-risk vessels
high_risk = risk[risk['risk_level'] == 'HIGH']
print(f"High-risk vessels: {len(high_risk)}")

# Link to simulation
simulation = pd.read_parquet('/path/to/simulation.parquet')
enriched = simulation.merge(risk[['mmsi', 'risk_level', 'risk_score']], on='mmsi')

# Analyze encounter patterns
encounter_stats = encounters.groupby('mmsi').size()
print(f"Vessels with encounters: {len(encounter_stats)}")
```

---

## Troubleshooting

### "GFW_API_TOKEN not found"
```bash
export GFW_API_TOKEN='your_token'
```

### "Identity file not found"
Run `enrich_vessel_identity.py` first - other scripts depend on it.

### Many "Not found in GFW"
Expected: 20-40% of simulated vessels may not exist in GFW (small vessels, new vessels).

### Rate limit errors (429)
Scripts auto-retry. If persistent, increase delay or run during off-peak hours.

---

## Next Steps

After enrichment:
1. **ML Training**: Use real risk labels with simulated tracks
2. **Anomaly Detection**: Compare simulated vs. real behavior
3. **Validation**: Check simulation realism
4. **Pattern Analysis**: Study sanctions evasion techniques
5. **Visualization**: Map high-risk areas, encounters, etc.

---

**Created**: November 6, 2025  
**Version**: 1.0.0  
**License**: Complies with GFW terms of service
