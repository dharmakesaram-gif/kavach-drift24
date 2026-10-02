import React, { useState } from 'react'

export default function CoCPage({ data }) {
  if (!data) return null;
  const lots = [...new Set(data.map(d => d.lot_id))].sort()
  const [selectedLot, setSelectedLot] = useState(lots[0] || 'LOT-ISRO-2026')
  const [isGenerating, setIsGenerating] = useState(false)

  const downloadCoC = async () => {
    setIsGenerating(true)
    try {
      const response = await fetch(`/api/v1/lots/${selectedLot}/coc`)
      if (!response.ok) throw new Error('Failed to generate CoC')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `CoC_${selectedLot}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert("Error generating Certificate of Conformance. Is the backend running?")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">📜 Certificate of Conformance</h2>
        <p className="text-muted text-sm mt-1">Generate Cryptographically signed PDF reports for flight-ready lots.</p>
      </div>

      <div className="bg-panel border border-edge rounded-xl p-8 shadow-xl">
        <div className="flex flex-col items-center justify-center space-y-6 py-8">
          <div className="w-24 h-24 rounded-full bg-accent/10 border-2 border-accent/20 flex items-center justify-center mb-2">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#7dd3fc" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <path d="M12 18v-6"></path>
              <path d="M9 15l3 3 3-3"></path>
            </svg>
          </div>
          
          <div className="text-center max-w-lg">
            <h3 className="text-xl font-display font-semibold text-white mb-2">Generate Aerospace-Grade Report</h3>
            <p className="text-gray-400 text-sm mb-6 leading-relaxed">
              Downloads a complete PDF incorporating AEC-Q001 summary statistics, the 168h drift 
              forecasting parameters, and individual accepted part lists. Stamped with SHA-256 for audit trails.
            </p>
          </div>

          <div className="flex items-center gap-4 bg-[#060a12] p-4 rounded-lg border border-edge w-full max-w-md">
            <label className="text-sm font-semibold text-gray-300 whitespace-nowrap">Target Lot:</label>
            <select 
              value={selectedLot} 
              onChange={e => setSelectedLot(e.target.value)}
              className="bg-panel border border-edge text-white text-sm rounded-md px-3 py-2 w-full focus:ring-1 focus:ring-accent outline-none"
            >
              {lots.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <button 
            onClick={downloadCoC}
            disabled={isGenerating}
            className={`mt-4 px-8 py-3 rounded-lg font-bold tracking-wide transition-all ${
              isGenerating 
                ? 'bg-accent/50 text-ink cursor-not-allowed' 
                : 'bg-accent text-ink hover:bg-accent/90 hover:scale-105 shadow-[0_0_20px_rgba(125,211,252,0.3)]'
            }`}
          >
            {isGenerating ? 'GENERATING PDF...' : 'DOWNLOAD CoC REPORT'}
          </button>
        </div>
      </div>
    </div>
  )
}
