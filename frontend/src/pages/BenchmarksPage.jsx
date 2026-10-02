import React from 'react'

export default function BenchmarksPage({ data }) {
  if (!data) return null;

  // Calculate baselines vs our model
  const total = data.length;
  const defects = data.filter(d => d.is_defect === 1).length;
  const nominal = total - defects;

  // 1. Static Datasheet Limit (Assuming 50uA)
  const staticTp = data.filter(d => d.is_defect === 1 && d.value_24h > 50).length;
  const staticFp = data.filter(d => d.is_defect === 0 && d.value_24h > 50).length;

  // 2. Global Z-Score (3.5 sigma across whole population, not per-lot)
  const mean24h = data.reduce((a, b) => a + (b.value_24h||0), 0) / total;
  const std24h = Math.sqrt(data.reduce((a, b) => a + Math.pow((b.value_24h||0) - mean24h, 2), 0) / total);
  const globalZLimit = mean24h + 3.5 * std24h;
  const globalTp = data.filter(d => d.is_defect === 1 && d.value_24h > globalZLimit).length;
  const globalFp = data.filter(d => d.is_defect === 0 && d.value_24h > globalZLimit).length;

  // 3. Our Ensembled Model (from final_decision)
  const ourTp = data.filter(d => d.is_defect === 1 && ['REJECT', 'REVIEW'].includes(d.final_decision)).length;
  const ourFp = data.filter(d => d.is_defect === 0 && ['REJECT', 'REVIEW'].includes(d.final_decision)).length;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">📈 Benchmarks & Baselines</h2>
        <p className="text-muted text-sm mt-1">Comparison of Kavach Drift24 vs standard aerospace screening methodologies.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <BaselineCard 
          title="Baseline: Static Limits" 
          method="Datasheet Max (50µA)"
          tp={staticTp} fp={staticFp} totalDefects={defects}
          color="border-reject"
        />
        <BaselineCard 
          title="Baseline: Global Z-Score" 
          method="Global Mean + 3.5σ"
          tp={globalTp} fp={globalFp} totalDefects={defects}
          color="border-review"
        />
        <BaselineCard 
          title="Kavach Ensembled AI" 
          method="Dynamic PAT + Drift Physics"
          tp={ourTp} fp={ourFp} totalDefects={defects}
          color="border-accept shadow-[0_0_15px_rgba(74,222,128,0.2)]"
        />
      </div>

      <div className="bg-panel border border-edge rounded-xl p-6 shadow-xl mt-8">
         <h3 className="text-lg font-display font-semibold text-white mb-4">Why Static Limits Fail for Space-Grade Parts</h3>
         <p className="text-sm text-gray-300 leading-relaxed mb-4">
           In high-reliability electronics, "maverick" components (latent defects) often exhibit initial parameter values that are 
           statistically abnormal for their specific wafer lot, yet still remain well beneath the manufacturer's absolute datasheet limit.
         </p>
         <ul className="list-disc pl-5 text-sm text-gray-400 space-y-2">
           <li><b>Static Limits:</b> Catch only catastrophic failures, resulting in high "escapes" into mission-critical hardware.</li>
           <li><b>Global Z-Scores:</b> Suffer from lot-to-lot manufacturing variance, resulting in high false scrap rates on intrinsically higher-leakage lots.</li>
           <li><b>Dynamic PAT (AEC-Q001):</b> Computes robust statistical limits strictly at the lot-level (using Median and Median Absolute Deviation) to identify relative mavericks.</li>
         </ul>
      </div>
    </div>
  )
}

function BaselineCard({ title, method, tp, fp, totalDefects, color }) {
  const recall = totalDefects > 0 ? (tp / totalDefects) * 100 : 0;
  const escapes = totalDefects - tp;
  
  return (
    <div className={`bg-panel border-2 rounded-xl p-6 shadow-lg ${color}`}>
      <h3 className="text-lg font-display font-bold text-white mb-1">{title}</h3>
      <p className="text-xs text-muted mb-6">{method}</p>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-xs mb-1 text-gray-300">
             <span>Recall (Defects Caught)</span>
             <span className="font-bold">{recall.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-[#060a12] rounded-full h-2 border border-edge">
             <div className="bg-accent h-full rounded-full" style={{ width: `${recall}%` }}></div>
          </div>
        </div>
        
        <div className="flex justify-between items-center border-t border-edge pt-4">
           <span className="text-sm text-gray-400">Escapes (FN)</span>
           <span className={`text-lg font-bold ${escapes > 0 ? 'text-reject' : 'text-accept'}`}>{escapes}</span>
        </div>
        <div className="flex justify-between items-center">
           <span className="text-sm text-gray-400">Scrap (FP)</span>
           <span className="text-lg font-bold text-white">{fp}</span>
        </div>
      </div>
    </div>
  )
}
