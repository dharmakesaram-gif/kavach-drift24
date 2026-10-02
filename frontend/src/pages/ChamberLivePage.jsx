import React, { useState, useEffect } from 'react'

export default function ChamberLivePage() {
  const [telemetry, setTelemetry] = useState(null)
  const [history, setHistory] = useState([])
  const [wsStatus, setWsStatus] = useState('CONNECTING')

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use window.location.host instead of hardcoded localhost
    // But for local dev (Vite on 5173), fallback to 8000
    const host = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
    const wsUrl = `${protocol}//${host}/ws/chamber-live`;
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => setWsStatus('CONNECTED')
    ws.onclose = () => setWsStatus('DISCONNECTED')
    ws.onerror = () => setWsStatus('ERROR')
    
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setTelemetry(data)
      setHistory(prev => {
        const newHist = [...prev, data]
        if (newHist.length > 50) newHist.shift()
        return newHist
      })
    }

    return () => ws.close()
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-white flex items-center gap-3">
          🔥 Live Chamber Telemetry
          <span className={`text-xs px-2 py-1 rounded font-bold tracking-widest ${
            wsStatus === 'CONNECTED' ? 'bg-accept/20 text-accept' : 'bg-reject/20 text-reject'
          }`}>
            {wsStatus}
          </span>
        </h2>
        <p className="text-muted text-sm mt-1">Real-time WebSocket feed from the 125°C Burn-In Thermal Chamber.</p>
      </div>

      {!telemetry ? (
        <div className="bg-panel border border-edge rounded-xl p-12 text-center text-muted">
          Waiting for telemetry... (Make sure FastAPI backend is running)
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-panel border border-edge rounded-xl p-6 shadow-xl relative overflow-hidden group">
            <div className={`absolute top-0 left-0 w-full h-1 ${telemetry.temperature_celsius > 130 ? 'bg-reject animate-pulse' : 'bg-accent'}`}></div>
            <div className="text-xs text-muted font-semibold uppercase tracking-wider mb-2">Temperature</div>
            <div className={`text-5xl font-display font-bold ${telemetry.temperature_celsius > 130 ? 'text-reject' : 'text-white'}`}>
              {telemetry.temperature_celsius?.toFixed(1)}<span className="text-2xl text-gray-500">°C</span>
            </div>
            <div className="text-xs text-gray-400 mt-2">Setpoint: 125.0°C (Limit: 132.0°C)</div>
          </div>
          
          <div className="bg-panel border border-edge rounded-xl p-6 shadow-xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-review"></div>
            <div className="text-xs text-muted font-semibold uppercase tracking-wider mb-2">Rack Current</div>
            <div className="text-5xl font-display font-bold text-white">
              {telemetry.rack_current_mA?.toFixed(0)}<span className="text-2xl text-gray-500">mA</span>
            </div>
            <div className="text-xs text-gray-400 mt-2">Limit: 2500 mA</div>
          </div>

          <div className="bg-panel border border-edge rounded-xl p-6 shadow-xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-purple-500"></div>
            <div className="text-xs text-muted font-semibold uppercase tracking-wider mb-2">Active Lot</div>
            <div className="text-2xl font-display font-bold text-white mt-2 truncate" title={telemetry.active_lot_id}>
              {telemetry.active_lot_id}
            </div>
            <div className="text-xs text-gray-400 mt-3">Targeting DB Ingestion</div>
          </div>

          <div className={`bg-panel border rounded-xl p-6 shadow-xl relative overflow-hidden group ${
            telemetry.interlock_tripped ? 'border-reject bg-reject/5' : 'border-edge'
          }`}>
            <div className={`absolute top-0 left-0 w-full h-1 ${telemetry.interlock_tripped ? 'bg-reject' : 'bg-accept'}`}></div>
            <div className="text-xs text-muted font-semibold uppercase tracking-wider mb-2">Safety Interlock</div>
            <div className={`text-2xl font-display font-bold mt-2 ${telemetry.interlock_tripped ? 'text-reject' : 'text-accept'}`}>
              {telemetry.interlock_tripped ? 'TRIPPED ⚠️' : 'ARMED ✅'}
            </div>
            <div className="text-xs text-gray-400 mt-3">{telemetry.interlock_reason || 'Nominal operation'}</div>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="bg-panel border border-edge rounded-xl p-4 shadow-xl overflow-hidden h-[300px] flex items-end gap-1 px-4">
           {history.map((h, i) => {
             const tempHeight = Math.max(0, Math.min(100, (h.temperature_celsius - 100) * 3)); // scale 100-133C to 0-100%
             const isWarning = h.temperature_celsius > 128;
             return (
               <div key={i} className="flex-1 flex flex-col justify-end group relative h-full">
                 <div 
                   className={`w-full rounded-t-sm transition-all duration-300 ${isWarning ? 'bg-reject' : 'bg-accent/70 group-hover:bg-accent'}`}
                   style={{ height: `${tempHeight}%` }}
                 ></div>
                 {/* Tooltip */}
                 <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-ink text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 whitespace-nowrap pointer-events-none z-10 border border-edge">
                   {h.temperature_celsius.toFixed(1)}°C<br/>
                   {h.rack_current_mA.toFixed(0)}mA
                 </div>
               </div>
             )
           })}
        </div>
      )}
    </div>
  )
}
