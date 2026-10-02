import React, { useState } from 'react'
import Plot from 'react-plotly.js'

export default function DistributionPage({ data }) {
  if (!data) return null;
  
  const lots = [...new Set(data.map(d => d.lot_id))].sort()
  const [selectedLot, setSelectedLot] = useState(lots[0])
  const [timePoint, setTimePoint] = useState('24h') // 0h or 24h
  
  const lotData = data.filter(d => d.lot_id === selectedLot)
  
  // Stats
  const med0h = lotData.reduce((acc, d) => acc + (d.value_0h||0), 0) / lotData.length;
  const med24h = lotData.reduce((acc, d) => acc + (d.value_24h||0), 0) / lotData.length;
  const flaggedCount = lotData.filter(d => ['REVIEW', 'REJECT'].includes(d.final_decision)).length;
  
  // Distribution calculations
  const valCol = timePoint === '24h' ? 'value_24h' : 'value_0h';
  const healthyVals = lotData.filter(d => d.is_defect === 0).map(d => d[valCol])
  const defectVals = lotData.filter(d => d.is_defect === 1).map(d => d[valCol])
  
  const medVal = timePoint === '24h' ? (lotData[0]?.lot_median_v24 || 10) : (lotData[0]?.lot_median_v0 || 10)
  const madVal = timePoint === '24h' ? (lotData[0]?.lot_mad_v24 || 1.5) : (lotData[0]?.lot_mad_v0 || 1.5)
  const dpatUpper = medVal + 3.5 * madVal
  const limitVal = lotData[0]?.datasheet_limit || 50.0
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">📊 Lot Distribution & Dynamic PAT</h2>
        <p className="text-muted text-sm mt-1">Inspect wafer lot distributions against static and AEC-Q001 DPAT limits.</p>
      </div>
      
      <div className="flex items-center gap-4 bg-panel border border-edge p-4 rounded-xl shadow-md">
        <div className="flex items-center gap-3">
          <label className="text-sm font-semibold text-gray-300">Select Lot:</label>
          <select 
            value={selectedLot} 
            onChange={e => setSelectedLot(e.target.value)}
            className="bg-[#0f172a] border border-edge text-white text-sm rounded-md px-3 py-1.5 focus:ring-1 focus:ring-accent outline-none"
          >
            {lots.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>
        <div className="h-6 w-px bg-edge mx-2"></div>
        <div className="flex gap-2">
          {['24h', '0h'].map(t => (
            <button key={t} onClick={() => setTimePoint(t)} className={`px-4 py-1.5 rounded-md text-xs font-semibold ${timePoint === t ? 'bg-accent text-ink' : 'bg-transparent text-gray-400 border border-edge hover:text-white'}`}>
              {t === '24h' ? '24h Burn-In' : '0h Pre-Burn-In'}
            </button>
          ))}
        </div>
      </div>
      
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-panel border border-edge rounded-lg p-4">
          <div className="text-xs text-muted mb-1">Lot Part Count</div>
          <div className="text-2xl font-display text-white">{lotData.length}</div>
        </div>
        <div className="bg-panel border border-edge rounded-lg p-4">
          <div className="text-xs text-muted mb-1">Median @ 0h</div>
          <div className="text-2xl font-display text-white">{med0h.toFixed(2)} µA</div>
        </div>
        <div className="bg-panel border border-edge rounded-lg p-4">
          <div className="text-xs text-muted mb-1">Median @ 24h</div>
          <div className="text-2xl font-display text-white flex items-baseline gap-2">
            {med24h.toFixed(2)} µA
            <span className="text-xs text-reject font-normal">+{((med24h - med0h)/med0h * 100).toFixed(1)}% drift</span>
          </div>
        </div>
        <div className="bg-panel border border-edge rounded-lg p-4">
          <div className="text-xs text-muted mb-1">Flagged Anomalies</div>
          <div className="text-2xl font-display text-reject">{flaggedCount}</div>
        </div>
      </div>
      
      <div className="bg-panel border border-edge rounded-xl shadow-xl overflow-hidden p-2">
        <Plot
          data={[
            { x: healthyVals, type: 'histogram', name: 'Healthy (Nominal)', marker: { color: '#3b82f6' }, opacity: 0.8 },
            ...(defectVals.length > 0 ? [{ x: defectVals, type: 'histogram', name: 'Latent Defect', marker: { color: '#ef4444' }, opacity: 0.9 }] : [])
          ]}
          layout={{
            title: `Parametric Distribution for ${selectedLot}`,
            barmode: 'overlay',
            height: 400,
            margin: { t: 50, b: 50, l: 50, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#e6edf7', family: 'IBM Plex Sans' },
            xaxis: { title: 'Leakage Current (µA)', gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Component Count', gridcolor: 'rgba(255,255,255,0.05)' },
            shapes: [
              { type: 'line', x0: limitVal, x1: limitVal, y0: 0, y1: 1, yref: 'paper', line: { color: '#dc2626', width: 2, dash: 'dash' } },
              { type: 'line', x0: dpatUpper, x1: dpatUpper, y0: 0, y1: 1, yref: 'paper', line: { color: '#f97316', width: 2, dash: 'dot' } },
              { type: 'line', x0: medVal, x1: medVal, y0: 0, y1: 1, yref: 'paper', line: { color: '#22c55e', width: 1.5, dash: 'solid' } }
            ],
            annotations: [
              { x: limitVal, y: 1.05, yref: 'paper', text: `Static Max (${limitVal.toFixed(1)})`, showarrow: false, font: { color: '#dc2626', size: 10 } },
              { x: dpatUpper, y: 0.9, yref: 'paper', text: `DPAT +3.5σ`, showarrow: false, font: { color: '#f97316', size: 10 }, xanchor: 'left', xshift: 5 },
              { x: medVal, y: 0.1, yref: 'paper', text: `Lot Median`, showarrow: false, font: { color: '#22c55e', size: 10 }, xanchor: 'right', xshift: -5 }
            ]
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  )
}
