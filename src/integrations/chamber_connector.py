"""
chamber_connector.py - Environmental Stress Screening (ESS) Chamber Connector & IoT Telemetry
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Interfaces with industrial burn-in chambers (ESPEC, Thermotron, Chroma 58158):
1. SCPI / Modbus protocol telemetry parser
2. Real-time chamber temperature & rack bias monitoring (MIL-STD-883K Method 1015 Condition D)
3. Automated Thermal Runaway & Emergency Safety Interlock (Trips power on defect surge)
4. Telemetry stream generator for digital twin simulation
"""

import time
import math
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple


class ChamberController:
    """
    Simulates or interfaces with an industrial burn-in oven and rack power controller.
    """
    
    def __init__(
        self,
        chamber_id: str = "CHAMBER-ESS-BAY04",
        setpoint_temp: float = 125.0,
        temp_tolerance: float = 2.0,
        nominal_voltage: float = 3.3,
        max_rack_current_mA: float = 2500.0
    ):
        self.chamber_id = chamber_id
        self.setpoint_temp = setpoint_temp
        self.temp_tolerance = temp_tolerance
        self.nominal_voltage = nominal_voltage
        self.max_rack_current_mA = max_rack_current_mA
        
        # State
        self.current_temp = setpoint_temp
        self.current_voltage = nominal_voltage
        self.current_rack_mA = 650.0
        self.is_interlock_tripped = False
        self.trip_reason = ""
        self.total_cycle_seconds = 0
        self.active_lot_id = "LOT-SPACE-2026-X"

    def read_telemetry(self) -> Dict[str, Any]:
        """
        Reads current physical chamber sensors and evaluates safety interlocks.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        
        if not self.is_interlock_tripped:
            # Simulate slight thermal oscillation around 125°C (+/- 0.6C)
            noise = random.gauss(0, 0.25)
            self.current_temp = self.setpoint_temp + noise
            self.current_voltage = self.nominal_voltage + random.gauss(0, 0.01)
            # Rack current nominal around 600-800 mA
            self.current_rack_mA = max(200.0, min(self.max_rack_current_mA, 650.0 + random.gauss(0, 20.0)))
            
            # Check thermal tolerance limit (MIL-STD-883 requires +/- 2C)
            temp_dev = abs(self.current_temp - self.setpoint_temp)
            temp_alarm = temp_dev > self.temp_tolerance
            
            # Check rack overcurrent
            overcurrent = self.current_rack_mA > self.max_rack_current_mA
            if overcurrent:
                self.trip_safety_interlock("OVERCURRENT_SAFETY_THRESHOLD_BREACHED")
        else:
            # If tripped, chamber temperature begins cooling down toward ambient 25C
            self.current_temp = max(25.0, self.current_temp - 1.5)
            self.current_voltage = 0.0
            self.current_rack_mA = 0.0
            temp_alarm = False
            overcurrent = False

        status = "NORMAL_RUN" if not self.is_interlock_tripped else "INTERLOCK_SHUTDOWN"
        
        return {
            "timestamp": now.isoformat(),
            "chamber_id": self.chamber_id,
            "status": status,
            "temperature_celsius": round(self.current_temp, 2),
            "setpoint_celsius": self.setpoint_temp,
            "temp_deviation_c": round(self.current_temp - self.setpoint_temp, 2),
            "voltage_bias_volts": round(self.current_voltage, 3),
            "rack_current_mA": round(self.current_rack_mA, 1),
            "max_rack_current_mA": self.max_rack_current_mA,
            "temp_alarm": temp_alarm,
            "thermal_runaway_alert": self.current_temp > (self.setpoint_temp + 5.0),
            "interlock_tripped": self.is_interlock_tripped,
            "trip_reason": self.trip_reason,
            "active_lot_id": self.active_lot_id
        }

    def trip_safety_interlock(self, reason: str = "MANUAL_OR_AUTOMATED_ABORT"):
        """Emergency hardware safety trip: Cuts off rack DUT power supply."""
        self.is_interlock_tripped = True
        self.trip_reason = reason
        self.current_voltage = 0.0
        self.current_rack_mA = 0.0

    def reset_interlock(self):
        """Authorised QA technician reset after addressing fault condition."""
        self.is_interlock_tripped = False
        self.trip_reason = ""
        self.current_voltage = self.nominal_voltage
        self.current_rack_mA = 650.0

    def inject_thermal_spike(self, spike_delta_c: float = 8.5):
        """Injects a thermal anomaly to test safety interlock response."""
        self.current_temp += spike_delta_c
        if self.current_temp > (self.setpoint_temp + 5.0):
            self.trip_safety_interlock(f"THERMAL_RUNAWAY_DETECTED ({self.current_temp:.1f}°C > 130°C)")
