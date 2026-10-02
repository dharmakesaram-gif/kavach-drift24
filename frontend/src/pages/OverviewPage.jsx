import React from 'react'
import Plot from 'react-plotly.js'

export default function OverviewPage({ data }) {
  if (!data) return null;

  const nParts = data.length;
  const nDefects = data.filter(d => d.is_defect === 1).length;
  const nCaught = data.filter(d => d.is_defect === 1 && ['REVIEW', 'REJECT'].includes(d.final_decision)).length;
  const escapes = nDefects - nCaught;
  const recall = nDefects > 0 ? (nCaught / nDefects) * 100 : 100.0;
  const earlyRejects = data.filter(d => d.early_rejection_24h).length;
  const hoursSaved = data.reduce((acc, curr) => acc + (curr.burnin_hours_saved || 0), 0);

  // Pie chart data
  const counts = { ACCEPT: 0, REVIEW: 0, REJECT: 0 }
  data.forEach(d => { if(counts[d.final_decision] !== undefined) counts[d.final_decision]++ })

  return (
    <div className="space-y-6">
      <div className="hero bg-gradient-to-br from-panel to-ink border border-edge rounded-xl p-8 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-accent via-purple-500 to-pink-500"></div>
        <div className="flex justify-between items-center z-10 relative">
          <div className="max-w-2xl">
            <h1 className="text-4xl font-display font-bold text-white mb-2">Aerospace Electronics Screening</h1>
            <p className="text-muted text-lg mb-6">Predictive latent-defect detection from 24-hour burn-in drift. SIH26170</p>
            <div className="flex gap-8 text-sm text-muted">
              <div><b className="block text-2xl text-white font-display">4.5×</b> lot median, yet under limit</div>
              <div><b className="block text-2xl text-white font-display">8.1σ</b> dynamic PAT score</div>
              <div><b className="block text-2xl text-white font-display">24 h</b> to reject, not 168 h</div>
            </div>
          </div>
          <div className="hidden lg:block opacity-80">
            <svg width="320" height="120" viewBox="0 0 320 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M10 100 L300 100" stroke="#1c2a44" strokeWidth="2" strokeDasharray="4 4" />
              <path d="M50 100 Q 150 20 280 100" stroke="#7dd3fc" strokeWidth="3" fill="none" />
              <circle cx="200" cy="40" r="6" fill="#f87171" className="animate-pulse" />
              <text x="215" y="45" fill="#f87171" fontSize="12" fontFamily="sans-serif">Latent Defect</text>
            </svg>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <MetricCard title="Total Components" value={nParts.toLocaleString()} sub={`${[...new Set(data.map(d=>d.lot_id))].length} Lots`} />
        <MetricCard title="Defect Interception" value={`${recall.toFixed(1)}%`} sub={`${nCaught}/${nDefects} Intercepted`} />
        <MetricCard title="Catastrophic Escapes" value={escapes} sub="Target: 0 Escapes" inverted={escapes > 0} />
        <MetricCard title="Early Terminations" value={earlyRejects} sub="Chamber freed @ 24h" />
        <MetricCard title="Chamber Hours Saved" value={`${hoursSaved.toLocaleString()}h`} sub="144h saved / part" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-panel border border-edge rounded-xl p-6">
          <h3 className="text-xl font-display font-semibold mb-4 text-white">⚙️ Multi-Layer Screening Architecture</h3>
          <p className="text-gray-300 text-sm mb-4 leading-relaxed">
            Standard static pass/fail testing fails because latent defects drift dynamically during 125°C Burn-In thermal stress while remaining numerically beneath datasheet upper bounds.
          </p>
          <ul className="space-y-2 text-sm text-gray-400 list-disc pl-5">
            <li><b>Data Layer:</b> Robust log-transform & empirical Bayes shrinkage for small lots.</li>
            <li><b>Module A:</b> 6-Layer Dynamic PAT fused via CalibratedStackedClassifier.</li>
            <li><b>Module B:</b> Forecasts 168h trajectory with Arrhenius physics quantile intervals.</li>
            <li><b>SPRT Sequential Decision:</b> ACCEPT/REJECT/UNCERTAIN at 24h based on Log-Likelihood Ratio.</li>
          </ul>
        </div>
        
        <div className="bg-panel border border-edge rounded-xl p-6 flex flex-col items-center justify-center">
          <h3 className="text-lg font-display font-semibold mb-2 text-white self-start w-full">Disposition Breakdown</h3>
          <Plot
            data={[{
              values: [counts.ACCEPT, counts.REVIEW, counts.REJECT],
              labels: ['ACCEPT', 'REVIEW', 'REJECT'],
              type: 'pie',
              hole: 0.65,
              marker: { colors: ['#4ade80', '#fbbf24', '#f87171'] },
              textinfo: 'percent',
              hoverinfo: 'label+value'
            }]}
            layout={{
              width: 280, height: 280,
              margin: { t: 10, b: 10, l: 10, r: 10 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              showlegend: false,
              font: { color: '#e6edf7', family: 'IBM Plex Sans' }
            }}
            config={{ displayModeBar: false }}
          />
          <div className="flex gap-4 text-xs font-semibold mt-2">
            <span className="text-accept">● ACCEPT</span>
            <span className="text-review">● REVIEW</span>
            <span className="text-reject">● REJECT</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ title, value, sub, inverted }) {
  return (
    <div className={`bg-panel border border-edge rounded-xl p-5 border-l-4 ${inverted ? 'border-l-reject' : 'border-l-accent'}`}>
      <p className="text-xs text-muted font-medium mb-1 truncate">{title}</p>
      <p className="text-3xl font-display font-bold text-white mb-1 tabular-nums">{value}</p>
      <p className={`text-xs ${inverted ? 'text-reject' : 'text-gray-400'}`}>{sub}</p>
    </div>
  )
}
