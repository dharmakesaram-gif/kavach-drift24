"""
test_integrations.py - Automated Unit & Integration Tests for Industrial Integrations
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Verifies:
1. ATE Parser (Teradyne J750 ASCII & Chroma CSV Datalogs)
2. Environmental Chamber Controller (Telemetry & Thermal Runaway Interlock)
3. MES Webhook Dispatcher (HMAC-SHA256 Signing & Event Schemas)
4. Aerospace Certificate of Conformance (PDF Generation & QR Code Seal)
5. Relational Database & Cryptographic Audit Trail (SHA-256 Record Hashes)
6. FastAPI REST Endpoints (Single-Part, Batch Lot, Telemetry, Health)
"""

import os
import unittest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.integrations.ate_parser import ATEDataParser
from src.integrations.chamber_connector import ChamberController
from src.integrations.mes_webhook import MESWebhookNotifier
from src.integrations.compliance_coc import CertificateOfConformanceGenerator
from src.integrations.database import ScreeningDatabase
from src.integrations.api import app, initialize_pipeline


class TestIndustrialIntegrations(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initializes API client, database, and pipeline state."""
        cls.client = TestClient(app)
        initialize_pipeline()
        cls.db = ScreeningDatabase(db_path=":memory:")

    def test_01_ate_parser_ascii_and_csv(self):
        """Tests ingestion of Teradyne J750 ASCII and Chroma CSV datalogs."""
        parser = ATEDataParser()
        raw_log = parser.generate_sample_ate_log(n_parts=30, lot_id="LOT-TEST-A1")
        
        parsed = parser.parse_ascii_ate_datalog(raw_log)
        df = parsed['dataframe']
        meta = parsed['metadata']
        
        self.assertEqual(meta['lot_id'], "LOT-TEST-A1")
        self.assertEqual(len(df), 30)
        self.assertIn('part_id', df.columns)
        self.assertIn('value_0h', df.columns)
        self.assertIn('value_24h', df.columns)
        self.assertTrue((df['value_0h'] > 0).all())
        
        # Test CSV parsing
        csv_str = df.to_csv(index=False)
        parsed_csv = parser.parse_csv_datalog(csv_str)
        self.assertEqual(len(parsed_csv['dataframe']), 30)

    def test_02_chamber_controller_and_safety_interlock(self):
        """Tests chamber temperature monitoring and emergency thermal trip."""
        chamber = ChamberController(setpoint_temp=125.0, temp_tolerance=2.0)
        
        # Normal reading
        t1 = chamber.read_telemetry()
        self.assertEqual(t1['status'], "NORMAL_RUN")
        self.assertAlmostEqual(t1['setpoint_celsius'], 125.0, delta=0.1)
        self.assertFalse(t1['interlock_tripped'])
        self.assertGreater(t1['voltage_bias_volts'], 3.0)
        
        # Inject thermal spike (runaway)
        chamber.inject_thermal_spike(12.0)
        t2 = chamber.read_telemetry()
        self.assertTrue(t2['interlock_tripped'])
        self.assertEqual(t2['status'], "INTERLOCK_SHUTDOWN")
        self.assertEqual(t2['voltage_bias_volts'], 0.0)  # DUT power cut
        
        # Reset interlock
        chamber.reset_interlock()
        t3 = chamber.read_telemetry()
        self.assertFalse(t3['interlock_tripped'])

    def test_03_mes_webhook_signing_and_payload(self):
        """Tests HMAC-SHA256 signature computation and Maverick Lot alert dispatch."""
        notifier = MESWebhookNotifier(hmac_secret="test-secret-2026")
        
        # Test signature
        sig = notifier.compute_signature(b'{"test": "payload"}')
        self.assertEqual(len(sig), 64)  # 64 hex characters for SHA-256
        
        # Dispatch Maverick Lot alert
        res = notifier.notify_maverick_lot("LOT-CRITICAL-99", rejection_rate_pct=8.4)
        self.assertEqual(res['event_type'], "LOT_MAVERICK_ALERT")
        self.assertTrue(res['signature'].startswith("sha256="))
        self.assertEqual(res['payload']['data']['rejection_rate_pct'], 8.4)

    def test_04_compliance_coc_pdf_generation(self):
        """Tests aerospace Certificate of Conformance PDF creation with QR seal."""
        gen = CertificateOfConformanceGenerator()
        
        lot_data = pd.DataFrame([
            {'part_id': f'PART_{i:02d}', 'value_0h': 10.0 + i*0.1, 'value_24h': 10.2 + i*0.1,
             'final_decision': 'ACCEPT' if i < 18 else 'REJECT', 'burnin_hours_saved': 0 if i < 18 else 144}
            for i in range(20)
        ])
        
        pdf_path = gen.generate_pdf("LOT-TEST-COC", lot_data)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)  # Valid non-empty PDF

    def test_05_database_and_audit_trail(self):
        """Tests persistence and cryptographic hash verification."""
        summary = {
            'median_0h': 10.1, 'median_24h': 10.3, 'sigma_0h': 0.4, 'sigma_24h': 0.5,
            'rejection_rate_pct': 2.5, 'maverick_lot_flag': False, 'hours_saved_total': 288
        }
        parts = [
            {'part_id': 'P01', 'value_0h': 10.0, 'value_24h': 10.2, 'final_decision': 'ACCEPT'},
            {'part_id': 'P02', 'value_0h': 10.5, 'value_24h': 42.0, 'final_decision': 'REJECT', 'hours_saved': 144}
        ]
        
        lot = self.db.record_lot_and_parts("LOT-AUDIT-01", summary, parts)
        self.assertEqual(lot.lot_id, "LOT-AUDIT-01")
        
        # Verify audit trail
        trail = self.db.get_recent_audit_trail(limit=5)
        self.assertGreater(len(trail), 0)
        self.assertEqual(len(trail[0]['record_hash_sha256']), 64)

    def test_06_fastapi_endpoints(self):
        """Tests REST endpoints for single part, batch lot, and telemetry."""
        # Health check
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], "OPERATIONAL")
        
        # Single Part Screening (Normal Part)
        p_normal = {
            "part_id": "SN-001",
            "lot_id": "LOT-FASTAPI-01",
            "value_0h": 10.2,
            "value_24h": 10.4,
            "datasheet_limit": 50.0
        }
        resp = self.client.post("/api/v1/screen/part", json=p_normal)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['final_decision'], "ACCEPT")
        self.assertIn("audit_sha256", data)
        
        # Single Part Screening (Latent Defect Part)
        p_defect = {
            "part_id": "SN-002",
            "lot_id": "LOT-FASTAPI-01",
            "value_0h": 10.5,
            "value_24h": 46.0,  # Severe latent drift
            "datasheet_limit": 50.0
        }
        resp = self.client.post("/api/v1/screen/part", json=p_defect)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data['final_decision'], ["REJECT", "REVIEW"])
        if data['final_decision'] == "REJECT":
            self.assertEqual(data['burnin_hours_saved'], 144)
            
        # Chamber Telemetry
        telem = {
            "chamber_id": "CHAMBER-ESS-BAY04",
            "temperature_celsius": 125.1,
            "voltage_bias_volts": 3.3,
            "rack_current_mA": 640.0,
            "active_lot_id": "LOT-FASTAPI-01"
        }
        resp = self.client.post("/api/v1/chamber/telemetry", json=telem)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], "NORMAL_RUN")
        
        # Audit logs endpoint
        resp = self.client.get("/api/v1/audit/logs")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)


if __name__ == "__main__":
    unittest.main()
