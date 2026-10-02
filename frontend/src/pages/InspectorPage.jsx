import React, { useState } from 'react'
import Plot from 'react-plotly.js'

export default function InspectorPage({ data }) {
  if (!data) return null;
  
  const flaggedIds = data.filter(d => ['REJECT', 'REVIEW'].includes(d.final_decision)).map(d => d.part_id)
  const [selectedPart, setSelectedPart] = useState(flaggedIds[0] || data[0].part_id)
  
  const partRow = data.find(d => d.part_id === selectedPart)
  if (!partRow) return null;
  
  const limitVal = partRow.datasheet_limit || 50.0;
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-display font-bold text-white">🔬 QA Inspector Deep-Dive</h2>
          <p className="text-muted text-sm mt-1">Transparent audit trail, natural language justification, and trajectory forecasting cone.</p>
        </div>
        <select 
          value={selectedPart} 
          onChange={e => setSelectedPart(e.target.value)}
          className="bg-[#0f172a] border border-edge text-white px-4 py-2 rounded-lg font-display font-medium focus:ring-1 focus:ring-accent outline-none"
        >
          {flaggedIds.map(id => <option key={id} value={id}>⚠️ {id}</option>)}
          {data.filter(d => !flaggedIds.includes(d.part_id)).slice(0,20).map(d => <option key={d.part_id} value={d.part_id}>✅ {d.part_id}</option>)}
        </select>
      </div>

      <div className={`p-6 rounded-xl border-l-4 bg-panel shadow-md border ${
        partRow.final_decision === 'REJECT' ? 'border-l-reject border-reject/20' : 
        partRow.final_decision === 'REVIEW' ? 'border-l-review border-review/20' : 'border-l-accept border-accept/20'
      }`}>
        <h3 className={`text-xl font-display font-bold mb-3 ${
          partRow.final_decision === 'REJECT' ? 'text-reject' : 
          partRow.final_decision === 'REVIEW' ? 'text-review' : 'text-accept'
        }`}>
          DISPOSITION: {partRow.final_decision} {partRow.final_decision === 'REJECT' ? '(Early Termination @ 24h)' : ''}
        </h3>
        <div className="bg-[#0f172a] p-4 rounded-lg border border-edge text-sm text-gray-300 leading-relaxed">
          <strong className="text-white">Reasoning:</strong> {partRow.primary_reason} <br/><br/>
          This component has an initial leakage of <b>{partRow.value_0h?.toFixed(2)}µA</b> which drifted to <b>{partRow.value_24h?.toFixed(2)}µA</b> after 24h. 
          Its composite risk score is <b>{(partRow.composite_risk_score||0).toFixed(3)}</b>, and its Dynamic PAT outlier score is <b>{partRow.z_pat_24h?.toFixed(2)}σ</b>.
          Our physics-informed Arrhenius drift model predicts it will reach <b>{partRow.pred_v168?.toFixed(2)}µA</b> by 168 hours, 
          {partRow.pred_v168 > limitVal ? <span className="text-reject"> EXCEEDING</span> : <span className="text-accept"> remaining below</span>} the {limitVal}µA datasheet limit.
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 bg-panel border border-edge rounded-xl shadow-xl overflow-hidden p-4">
          <h4 className="text-sm font-semibold text-white mb-2">📈 Multi-Point Aging Trajectory & Forecast Cone</h4>
          <Plot
            data={[
              { 
                x: [0, 168], y: [limitVal, limitVal], 
                type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2, dash: 'dot' }, name: 'Datasheet Max' 
              },
              { 
                x: [0, 24], y: [partRow.value_0h, partRow.value_24h], 
                type: 'scatter', mode: 'lines+markers', line: { color: '#e11d48', width: 3 }, marker: { size: 10 }, name: 'Observed (0h-24h)' 
              },
              { 
                x: [24, 168], y: [partRow.value_24h, partRow.pred_v168], 
                type: 'scatter', mode: 'lines+markers', line: { color: '#f97316', width: 2.5, dash: 'dash' }, marker: { size: 8, symbol: 'diamond' }, name: 'Projected 168h' 
              },
              {
                x: [24, 168, 168, 24], y: [partRow.value_24h, partRow.pred_v168_upper, partRow.pred_v168*0.9, partRow.value_24h],
                fill: 'toself', fillcolor: 'rgba(239, 68, 68, 0.15)', line: { color: 'transparent' }, name: 'Confidence Cone'
              }
            ]}
            layout={{
              height: 380,
              margin: { t: 20, b: 40, l: 40, r: 20 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#e6edf7', family: 'IBM Plex Sans' },
              xaxis: { title: 'Burn-In Stress Duration (Hours @ 125°C)', gridcolor: 'rgba(255,255,255,0.05)' },
              yaxis: { title: 'Leakage (µA)', gridcolor: 'rgba(255,255,255,0.05)' },
              hovermode: 'x unified',
              legend: { orientation: 'h', y: 1.1 }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="bg-panel border border-edge rounded-xl shadow-xl overflow-hidden p-4">
           <h4 className="text-sm font-semibold text-white mb-2">🔬 Parametric Attributes</h4>
           <div className="space-y-3 mt-4">
             <div className="flex justify-between items-center border-b border-edge pb-2">
               <span className="text-sm text-gray-400">Risk Score</span>
               <span className="text-sm font-bold text-white">{partRow.composite_risk_score?.toFixed(3)}</span>
             </div>
             <div className="flex justify-between items-center border-b border-edge pb-2">
               <span className="text-sm text-gray-400">Anomaly Mod A</span>
               <span className="text-sm font-bold text-white">{partRow.anomaly_score_mod_a?.toFixed(3)}</span>
             </div>
             <div className="flex justify-between items-center border-b border-edge pb-2">
               <span className="text-sm text-gray-400">DPAT 0h Z-Score</span>
               <span className="text-sm font-bold text-white">{partRow.z_pat_0h?.toFixed(2)}σ</span>
             </div>
             <div className="flex justify-between items-center border-b border-edge pb-2">
               <span className="text-sm text-gray-400">DPAT 24h Z-Score</span>
               <span className="text-sm font-bold text-white">{partRow.z_pat_24h?.toFixed(2)}σ</span>
             </div>
             <div className="flex justify-between items-center pb-2">
               <span className="text-sm text-gray-400">Drift Z-Score</span>
               <span className="text-sm font-bold text-white">{partRow.z_drift_24h?.toFixed(2)}σ</span>
             </div>
           </div>
        </div>
      </div>
    </div>
  )
}
