#!/usr/bin/env python3
"""
Risk Assessment Enrichment Dataset Generator

Creates a comprehensive risk assessment dataset combining all event types.
Calculates risk scores based on suspicious behavior patterns.

Output: enrichment_outputs/risk_assessment_enrichment.csv
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.analyzer import VesselAnalyzer
from src.config import Config

async def enrich_risk_assessment(identity_csv: str, output_csv: str, days_back: int = 365):
    print("=" * 80)
    print("RISK ASSESSMENT ENRICHMENT")
    print("=" * 80)
    print()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading vessel identity data from: {identity_csv}")
    try:
        identity_df = pd.read_csv(identity_csv)
    except FileNotFoundError:
        print(f"❌ ERROR: Identity file not found: {identity_csv}")
        return

    matched_vessels = identity_df[identity_df['matched'] == True].copy()
    print(f"Found {len(matched_vessels)} matched vessels in GFW")
    print(f"Analyzing risk for {days_back} days back")
    print()

    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found")
        return

    client = GFWClient()
    analyzer = VesselAnalyzer(client)

    all_risk_assessments = []
    start_time = datetime.now()

    for idx, row in matched_vessels.iterrows():
        vessel_id = row['gfw_vessel_id']
        vessel_name = row['simulated_vessel_name']
        mmsi = row['mmsi']

        elapsed = (datetime.now() - start_time).total_seconds()
        if idx > 0:
            eta_mins = (elapsed / (idx + 1)) * (len(matched_vessels) - idx - 1) / 60
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name} [ETA: {eta_mins:.1f}m]...", end=" ")
        else:
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name}...", end=" ")

        try:
            # Full vessel analysis with risk assessment
            result = await analyzer.analyze_vessel(vessel_id, days_back=days_back)

            if result and result.risk_assessment:
                risk = result.risk_assessment
                events = result.events

                all_risk_assessments.append({
                    'mmsi': mmsi,
                    'simulated_vessel_name': vessel_name,
                    'risk_level': risk.risk_level,
                    'risk_score': risk.total_risk_score,
                    'encounters_count': len(events.get('encounters', [])),
                    'gaps_count': len(events.get('gaps', [])),
                    'loitering_count': len(events.get('loitering', [])),
                    'port_visits_count': len(events.get('port_visits', [])),
                    'ais_coverage_pct': result.coverage_percentage,
                    'risk_factor_sts': risk.risk_factors.get('STS Transfers', 0),
                    'risk_factor_gaps': risk.risk_factors.get('Dark Periods', 0),
                    'risk_factor_loitering': risk.risk_factors.get('Loitering Events', 0),
                    'risk_factor_coverage': risk.risk_factors.get('Low AIS Coverage', 0),
                    'risk_summary': risk.summary,
                    'analysis_period_days': result.analysis_period_days,
                    'assessment_timestamp': datetime.utcnow().isoformat()
                })

                print(f"✅ {risk.risk_level} (score: {risk.total_risk_score})")
            else:
                print("⭕ No data")

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:50]}")

        await asyncio.sleep(0.5)

    risk_df = pd.DataFrame(all_risk_assessments)
    if len(risk_df) > 0:
        risk_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("RISK ASSESSMENT ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Vessels processed: {len(matched_vessels)}")
    print(f"Assessments generated: {len(risk_df)}")
    print()
    if len(risk_df) > 0:
        print("Risk level distribution:")
        print(risk_df['risk_level'].value_counts().to_string())
    print()
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(risk_df)} rows")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enrich risk assessment from GFW')
    parser.add_argument('--identity', type=str, default='enrichment_outputs/vessel_identity_enrichment.csv')
    parser.add_argument('--output', type=str, default='enrichment_outputs/risk_assessment_enrichment.csv')
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(enrich_risk_assessment(args.identity, args.output, args.days))
