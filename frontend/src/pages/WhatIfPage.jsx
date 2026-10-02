import React, { useState, useMemo } from 'react'

export default function WhatIfPage({ data }) {
  const [zPatThreshold, setZPatThreshold] = useState(3.5)
  const [safetyK, setSafetyK] = useState(2.5)
  const [fnCost, setFnCost] = useState(50)
  const [fpCost, setFpCost] = useState(1)

  if (!data) return null;

  const simResults = useMemo(() => {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    data.forEach(d => {
      // Simulate decision
      const patFlag = d.z_pat_0h > zPatThreshold || d.z_pat_24h > zPatThreshold;
      // Rough approximation of safety slope limit breach based on K
      const slopeLimitBreach = d.z_drift_24h > safetyK;
      const anomalyFlag = d.anomaly_score_mod_a >= 0.45;
      
      const isFlagged = patFlag || slopeLimitBreach || anomalyFlag;
      const isDefect = d.is_defect === 1;
      
      if (isFlagged && isDefect) tp++;
      else if (!isFlagged && !isDefect) tn++;
      else if (isFlagged && !isDefect) fp++;
      else if (!isFlagged && isDefect) fn++;
    });
    
    const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 100;
    const cost = (fnCost * fn) + (fpCost * fp);
    const hoursSaved = tp * 144;
    
    return { tp, tn, fp, fn, recall, cost, hoursSaved }
  }, [data, zPatThreshold, safetyK, fnCost, fpCost])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">🎛️ Interactive What-If Simulation</h2>
        <p className="text-muted text-sm mt-1">Fine-tune screening sensitivity sliders in real time and see the impact.</p>
      </div>

      <div className="grid grid-cols-2 gap-8">
        <div className="bg-panel border border-edge rounded-xl p-6 space-y-6">
          <h3 className="text-lg font-display font-semibold text-white border-b border-edge pb-2">Algorithm Thresholds</h3>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-300">Dynamic PAT Z-Score (σ)</span>
              <span className="text-accent font-bold">{zPatThreshold.toFixed(1)}</span>
            </div>
            <input 
              type="range" min="2.0" max="4.5" step="0.1" 
              value={zPatThreshold} onChange={e => setZPatThreshold(parseFloat(e.target.value))}
              className="w-full accent-accent"
            />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-300">Safety Slope Multiplier (k)</span>
              <span className="text-accent font-bold">{safetyK.toFixed(1)}</span>
            </div>
            <input 
              type="range" min="1.5" max="4.0" step="0.1" 
              value={safetyK} onChange={e => setSafetyK(parseFloat(e.target.value))}
              className="w-full accent-accent"
            />
          </div>
        </div>

        <div className="bg-panel border border-edge rounded-xl p-6 space-y-6">
          <h3 className="text-lg font-display font-semibold text-white border-b border-edge pb-2">Business Cost Matrix</h3>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-300">Escape Penalty Weight (FN)</span>
              <span className="text-reject font-bold">{fnCost}</span>
            </div>
            <input 
              type="range" min="10" max="100" step="5" 
              value={fnCost} onChange={e => setFnCost(parseInt(e.target.value))}
              className="w-full accent-reject"
            />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-300">Scrap Penalty Weight (FP)</span>
              <span className="text-review font-bold">{fpCost}</span>
            </div>
            <input 
              type="range" min="1" max="10" step="1" 
              value={fpCost} onChange={e => setFpCost(parseInt(e.target.value))}
              className="w-full accent-review"
            />
          </div>
        </div>
      </div>

      <div className="bg-ink border border-accent/20 rounded-xl p-6">
        <h3 className="text-lg font-display font-semibold text-accent mb-4">📊 Real-Time Simulated Performance</h3>
        
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-panel border border-edge rounded-lg p-4">
            <div className="text-xs text-muted mb-1">Defect Recall</div>
            <div className="text-3xl font-display text-white">{simResults.recall.toFixed(1)}%</div>
            <div className="text-xs text-accept mt-1">{simResults.tp}/{simResults.tp+simResults.fn} Caught</div>
          </div>
          
          <div className="bg-panel border border-edge rounded-lg p-4 border-b-2 border-b-reject">
            <div className="text-xs text-muted mb-1">Escapes (FN)</div>
            <div className="text-3xl font-display text-reject">{simResults.fn}</div>
            <div className="text-xs text-gray-400 mt-1">Missed defects</div>
          </div>
          
          <div className="bg-panel border border-edge rounded-lg p-4">
            <div className="text-xs text-muted mb-1">Scrap (FP)</div>
            <div className="text-3xl font-display text-white">{simResults.fp}</div>
            <div className="text-xs text-gray-400 mt-1">False alarms</div>
          </div>
          
          <div className="bg-panel border border-edge rounded-lg p-4 border-b-2 border-b-accent">
            <div className="text-xs text-muted mb-1">Simulated Cost Score</div>
            <div className="text-3xl font-display text-accent">{simResults.cost}</div>
            <div className="text-xs text-gray-400 mt-1">Lower is better</div>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-accept/10 border border-accept/30 rounded-lg text-accept flex items-center gap-2">
          <span>✅ Estimated Thermal Chamber Hours Saved: <b>{simResults.hoursSaved.toLocaleString()} hours</b> by terminating early at 24h.</span>
        </div>
      </div>
    </div>
  )
}
