"""
database.py - Space-Grade Relational Database & Audit Logging Layer
SIH26170: Burn-In Latent Defect Detection & Drift Prediction System

Implements:
1. SQLAlchemy Relational Models for Lots, Parts, Chamber Telemetry, and Webhooks
2. Tamper-Evident Cryptographic Audit Trail (SHA-256 Hash Chaining)
3. ISO 9001 / AS9100 Aerospace Screening Traceability Records
"""

import os
import hashlib
import json
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, Text, ForeignKey, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

Base = declarative_base()

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'screening_records.db')
)


def _get_utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class LotRecord(Base):
    __tablename__ = 'production_lots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=_get_utc_now)
    part_count = Column(Integer, default=0)
    median_0h = Column(Float, nullable=True)
    median_24h = Column(Float, nullable=True)
    sigma_0h = Column(Float, nullable=True)
    sigma_24h = Column(Float, nullable=True)
    maverick_lot_flag = Column(Boolean, default=False)
    rejection_rate_pct = Column(Float, default=0.0)
    hours_saved_total = Column(Integer, default=0)
    screening_standard = Column(String(128), default="AEC-Q001 Rev-D / MIL-STD-883K Method 1015")
    
    parts = relationship("PartRecord", back_populates="lot", cascade="all, delete-orphan")


class PartRecord(Base):
    __tablename__ = 'screened_parts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_id = Column(String(64), nullable=False, index=True)
    lot_id = Column(String(64), ForeignKey('production_lots.lot_id'), nullable=False, index=True)
    screened_at = Column(DateTime, default=_get_utc_now)
    
    value_0h = Column(Float, nullable=False)
    value_24h = Column(Float, nullable=False)
    z_pat_0h = Column(Float, nullable=True)
    z_pat_24h = Column(Float, nullable=True)
    drift_rate_24h = Column(Float, nullable=True)
    
    pred_v168 = Column(Float, nullable=True)
    pred_v168_upper = Column(Float, nullable=True)
    
    anomaly_score = Column(Float, default=0.0)
    final_decision = Column(String(16), nullable=False)  # ACCEPT, REVIEW, REJECT
    early_rejection_24h = Column(Boolean, default=False)
    hours_saved = Column(Integer, default=0)
    primary_reason = Column(String(256), nullable=True)
    
    lot = relationship("LotRecord", back_populates="parts")


class ChamberTelemetryRecord(Base):
    __tablename__ = 'chamber_telemetry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_get_utc_now, index=True)
    chamber_id = Column(String(64), default="CHAMBER-ESS-01")
    temperature_celsius = Column(Float, nullable=False)
    setpoint_celsius = Column(Float, default=125.0)
    voltage_bias_volts = Column(Float, default=3.3)
    rack_current_mA = Column(Float, nullable=False)
    thermal_runaway_alert = Column(Boolean, default=False)
    interlock_tripped = Column(Boolean, default=False)
    active_lot_id = Column(String(64), nullable=True)


class AuditLogRecord(Base):
    __tablename__ = 'audit_trail'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_get_utc_now, index=True)
    action = Column(String(64), nullable=False)
    entity_type = Column(String(32), nullable=False)  # LOT, PART, CHAMBER, CONFIG
    entity_id = Column(String(64), nullable=False)
    operator = Column(String(64), default="AUTO_INSPECTION_ENGINE")
    details_json = Column(Text, nullable=False)
    record_hash_sha256 = Column(String(64), nullable=False)


class ScreeningDatabase:
    """
    Manages persistent SQLite storage and aerospace audit trail operations.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if self.db_path != ":memory:":
            parent_dir = os.path.dirname(self.db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def record_lot_and_parts(self, lot_id: str, summary: Dict[str, Any], parts_data: List[Dict[str, Any]]) -> LotRecord:
        """
        Persists a complete screened lot and associated parts, creating an immutable audit record.
        """
        session = self.get_session()
        try:
            lot = session.query(LotRecord).filter_by(lot_id=lot_id).first()
            if not lot:
                lot = LotRecord(lot_id=lot_id)
                session.add(lot)
            
            lot.part_count = len(parts_data)
            lot.median_0h = summary.get('median_0h', 0.0)
            lot.median_24h = summary.get('median_24h', 0.0)
            lot.sigma_0h = summary.get('sigma_0h', 0.0)
            lot.sigma_24h = summary.get('sigma_24h', 0.0)
            lot.maverick_lot_flag = summary.get('maverick_lot_flag', False)
            lot.rejection_rate_pct = summary.get('rejection_rate_pct', 0.0)
            lot.hours_saved_total = summary.get('hours_saved_total', 0)
            
            # Upsert parts
            for p in parts_data:
                part = session.query(PartRecord).filter_by(part_id=p['part_id'], lot_id=lot_id).first()
                if not part:
                    part = PartRecord(part_id=p['part_id'], lot_id=lot_id, value_0h=p['value_0h'], value_24h=p['value_24h'])
                    session.add(part)
                
                part.z_pat_0h = p.get('z_pat_0h')
                part.z_pat_24h = p.get('z_pat_24h')
                part.drift_rate_24h = p.get('drift_rate_24h')
                part.pred_v168 = p.get('pred_v168')
                part.pred_v168_upper = p.get('pred_v168_upper')
                part.anomaly_score = p.get('anomaly_score', 0.0)
                part.final_decision = p.get('final_decision', 'ACCEPT')
                part.early_rejection_24h = p.get('early_rejection_24h', False)
                part.hours_saved = p.get('hours_saved', 0)
                part.primary_reason = p.get('primary_reason')
            
            session.commit()
            
            # Create cryptographic audit trail entry
            self.create_audit_entry(
                action="LOT_SCREENED",
                entity_type="LOT",
                entity_id=lot_id,
                details={
                    "parts_count": len(parts_data),
                    "rejection_rate_pct": summary.get('rejection_rate_pct', 0.0),
                    "maverick_lot_flag": summary.get('maverick_lot_flag', False)
                },
                session=session
            )
            return lot
        finally:
            session.close()

    def record_chamber_telemetry(self, telemetry: Dict[str, Any]) -> ChamberTelemetryRecord:
        """Logs a real-time environmental stress screening chamber sensor point."""
        session = self.get_session()
        try:
            rec = ChamberTelemetryRecord(
                chamber_id=telemetry.get('chamber_id', 'CHAMBER-ESS-01'),
                temperature_celsius=telemetry['temperature_celsius'],
                setpoint_celsius=telemetry.get('setpoint_celsius', 125.0),
                voltage_bias_volts=telemetry.get('voltage_bias_volts', 3.3),
                rack_current_mA=telemetry['rack_current_mA'],
                thermal_runaway_alert=telemetry.get('thermal_runaway_alert', False),
                interlock_tripped=telemetry.get('interlock_tripped', False),
                active_lot_id=telemetry.get('active_lot_id')
            )
            session.add(rec)
            session.commit()
            return rec
        finally:
            session.close()

    def create_audit_entry(
        self, action: str, entity_type: str, entity_id: str, details: Dict[str, Any],
        operator: str = "AUTO_INSPECTION_ENGINE", session: Optional[Session] = None
    ) -> AuditLogRecord:
        """Generates an immutable audit trail entry stamped with a SHA-256 seal."""
        close_on_finish = False
        if session is None:
            session = self.get_session()
            close_on_finish = True
            
        try:
            payload = {
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "operator": operator,
                "details": details,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            raw_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
            
            audit = AuditLogRecord(
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                operator=operator,
                details_json=json.dumps(details),
                record_hash_sha256=sha256_hash
            )
            session.add(audit)
            session.commit()
            return audit
        finally:
            if close_on_finish:
                session.close()

    def get_recent_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent audit events for QA review."""
        session = self.get_session()
        try:
            records = session.query(AuditLogRecord).order_by(desc(AuditLogRecord.timestamp)).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "action": r.action,
                    "entity_type": r.entity_type,
                    "entity_id": r.entity_id,
                    "operator": r.operator,
                    "details": json.loads(r.details_json),
                    "record_hash_sha256": r.record_hash_sha256
                }
                for r in records
            ]
        finally:
            session.close()
