import React from 'react'

export default function SPRTPage({ data }) {
  if (!data) return null;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">🧬 SPRT Sequential Decision Engine</h2>
        <p className="text-muted text-sm mt-1">Wald's Sequential Probability Ratio Test (SPRT) logic for early burn-in termination.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-panel border border-edge rounded-xl p-8 shadow-xl">
          <h3 className="text-lg font-display font-semibold text-white mb-4 border-b border-edge pb-2">Log-Likelihood Ratio Formulation</h3>
          <p className="text-sm text-gray-300 mb-4 leading-relaxed">
            The decision engine continuously evaluates the Log-Likelihood Ratio <code className="bg-ink px-1 rounded text-accent">Λ(n)</code> of a 
            component belonging to the "Latent Defect" distribution vs. the "Nominal" distribution as hourly telemetry is ingested.
          </p>
          <div className="bg-[#0f172a] p-4 rounded-lg font-mono text-xs text-accent border border-edge mb-4">
            Λ(n) = Σ [ ln( P(x_i | θ_defect) / P(x_i | θ_nominal) ) ]
          </div>
          <ul className="text-sm text-gray-400 space-y-2 list-disc pl-5">
            <li><strong className="text-white">Upper Boundary (A):</strong> Pre-calculated based on Target Escapes. Breaching triggers immediate <b>REJECT</b>.</li>
            <li><strong className="text-white">Lower Boundary (B):</strong> Pre-calculated based on Acceptable False Scrap. Breaching triggers immediate <b>ACCEPT</b>.</li>
            <li><strong className="text-white">Uncertain Region:</strong> Between boundaries, testing continues to next checkpoint.</li>
          </ul>
        </div>

        <div className="bg-panel border border-edge rounded-xl p-8 shadow-xl flex flex-col justify-center">
          <h3 className="text-lg font-display font-semibold text-white mb-4">Economics of Early Termination</h3>
          <div className="space-y-4">
             <div className="flex justify-between items-center bg-[#060a12] p-4 rounded-lg border border-edge">
                <span className="text-sm text-gray-400">Total Simulated Parts</span>
                <span className="text-xl font-bold text-white">{data.length}</span>
             </div>
             <div className="flex justify-between items-center bg-accept/10 p-4 rounded-lg border border-accept/30">
                <span className="text-sm text-accept font-semibold">Standard Burn-In Duration</span>
                <span className="text-xl font-bold text-accept">168 Hours</span>
             </div>
             <div className="flex justify-between items-center bg-accent/10 p-4 rounded-lg border border-accent/30">
                <span className="text-sm text-accent font-semibold">Average SPRT Decision Time</span>
                <span className="text-xl font-bold text-accent">24 Hours</span>
             </div>
             <div className="mt-4 text-xs text-muted">
                * Note: The current pipeline evaluates at the 24h milestone. Sub-24h evaluations require higher-frequency chamber readouts.
             </div>
          </div>
        </div>
      </div>
    </div>
  )
}
