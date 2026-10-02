import React, { useState } from 'react'

export default function FlaggedPage({ data }) {
  const [filter, setFilter] = useState('ALL')
  if (!data) return null;

  let filtered = data;
  if (filter === 'REJECT') filtered = data.filter(d => d.final_decision === 'REJECT')
  if (filter === 'REVIEW') filtered = data.filter(d => d.final_decision === 'REVIEW')
  if (filter === 'ACCEPT') filtered = data.filter(d => d.final_decision === 'ACCEPT')

  // Sort by risk score
  filtered.sort((a, b) => (b.composite_risk_score || 0) - (a.composite_risk_score || 0))

  return (
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">⚠️ Flagged Components</h2>
        <p className="text-muted text-sm mt-1">Filter and review components flagged by Module A (Dynamic PAT) and Module B (Drift).</p>
      </div>

      <div className="flex gap-4 mb-4">
        {['ALL', 'REJECT', 'REVIEW', 'ACCEPT'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider transition-colors border ${
              filter === f 
                ? 'bg-accent/20 border-accent text-accent' 
                : 'bg-transparent border-edge text-gray-400 hover:border-gray-500 hover:text-white'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="bg-panel border border-edge rounded-xl shadow-lg flex-1 overflow-hidden flex flex-col">
        <div className="overflow-auto flex-1">
          <table className="w-full text-left border-collapse text-sm whitespace-nowrap">
            <thead className="bg-[#0f172a] sticky top-0 z-10 border-b border-edge">
              <tr>
                <th className="p-4 font-semibold text-gray-300">Part ID</th>
                <th className="p-4 font-semibold text-gray-300">Lot ID</th>
                <th className="p-4 font-semibold text-gray-300">Disposition</th>
                <th className="p-4 font-semibold text-gray-300 text-right">Risk Score</th>
                <th className="p-4 font-semibold text-gray-300 text-right">0h (µA)</th>
                <th className="p-4 font-semibold text-gray-300 text-right">24h (µA)</th>
                <th className="p-4 font-semibold text-gray-300 text-right">Proj 168h (µA)</th>
                <th className="p-4 font-semibold text-gray-300 text-right">DPAT (σ)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-edge">
              {filtered.map(row => (
                <tr key={row.part_id} className="hover:bg-white/5 transition-colors group">
                  <td className="p-4 font-medium text-gray-200">{row.part_id}</td>
                  <td className="p-4 text-muted">{row.lot_id}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      row.final_decision === 'REJECT' ? 'bg-reject/20 text-reject border border-reject/30' :
                      row.final_decision === 'REVIEW' ? 'bg-review/20 text-review border border-review/30' :
                      'bg-accept/10 text-accept'
                    }`}>
                      {row.final_decision}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <span className={row.composite_risk_score > 0.6 ? 'text-reject font-bold' : 'text-gray-300'}>
                      {row.composite_risk_score?.toFixed(2) || '0.00'}
                    </span>
                  </td>
                  <td className="p-4 text-right text-gray-400">{row.value_0h?.toFixed(2)}</td>
                  <td className="p-4 text-right text-white">{row.value_24h?.toFixed(2)}</td>
                  <td className="p-4 text-right">
                    <span className={row.pred_v168 > 50.0 ? 'text-reject font-semibold' : 'text-gray-300'}>
                      {row.pred_v168?.toFixed(2)}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <span className={row.z_pat_24h > 3.5 ? 'text-[#f97316] font-semibold' : 'text-gray-400'}>
                      {row.z_pat_24h?.toFixed(1)}σ
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="p-3 border-t border-edge bg-ink/50 text-xs text-muted flex justify-between">
          <span>Showing {filtered.length} components</span>
          <span>Target: 0 Escapes</span>
        </div>
      </div>
    </div>
  )
}
