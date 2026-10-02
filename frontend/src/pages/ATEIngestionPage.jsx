import React, { useState, useRef } from 'react'

export default function ATEIngestionPage() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/v1/ingest/ate', {
        method: 'POST',
        body: formData,
      })
      
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">🗄️ Raw ATE Datalog Ingestion</h2>
        <p className="text-muted text-sm mt-1">Upload raw CSV/ASCII logs from Automated Test Equipment for batch processing.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-panel border border-edge rounded-xl p-8 shadow-xl flex flex-col items-center justify-center border-dashed">
          <div className="w-16 h-16 rounded-full bg-[#0f172a] border border-edge flex items-center justify-center mb-4">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8aa0c0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          <h3 className="text-white font-semibold mb-2">Select Datalog File</h3>
          <p className="text-xs text-muted text-center mb-6">Supports .txt, .csv formats with space/comma delimited parametric columns.</p>
          
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept=".txt,.csv"
            onChange={e => setFile(e.target.files[0])}
          />
          
          <div className="flex gap-4 w-full">
            <button 
              onClick={() => fileInputRef.current.click()}
              className="flex-1 py-2 rounded-lg bg-[#0f172a] border border-edge text-gray-300 hover:text-white hover:border-gray-500 transition-colors text-sm font-semibold"
            >
              {file ? file.name : 'BROWSE FILES'}
            </button>
            <button 
              onClick={handleUpload}
              disabled={!file || loading}
              className={`flex-1 py-2 rounded-lg font-bold text-sm tracking-wide transition-all ${
                !file || loading
                  ? 'bg-accent/20 text-accent/50 cursor-not-allowed' 
                  : 'bg-accent text-ink hover:bg-accent/90'
              }`}
            >
              {loading ? 'PROCESSING...' : 'UPLOAD & SCREEN'}
            </button>
          </div>
          {error && <p className="text-reject text-xs mt-4">Error: {error}</p>}
        </div>

        <div className="bg-[#0f172a] border border-edge rounded-xl p-6 shadow-inner">
          <h3 className="text-sm font-semibold text-white mb-4 border-b border-edge pb-2">Sample ATE Format</h3>
          <pre className="text-xs text-gray-400 font-mono overflow-x-auto">
{`Lot_ID,Part_ID,Test_Type,Voltage,Current_0h,Current_24h,Temp
LOT-DEMO-A1,PART-001,ILEAK,3.3V,10.5,10.6,125C
LOT-DEMO-A1,PART-002,ILEAK,3.3V,11.0,18.5,125C
LOT-DEMO-A1,PART-003,ILEAK,3.3V,9.8,9.9,125C
LOT-DEMO-A1,PART-004,ILEAK,3.3V,10.2,42.1,125C`}
          </pre>
          <div className="mt-4 text-xs text-muted">
            The parser automatically maps common column headers (e.g., Current_0h -&gt; value_0h) and batches them to the screening engine.
          </div>
        </div>
      </div>

      {results && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h3 className="text-xl font-display font-bold text-white mb-4">Ingestion Results</h3>
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-panel border border-edge rounded-lg p-4">
              <div className="text-xs text-muted mb-1">Total Screened</div>
              <div className="text-3xl font-display text-white">{results.total_screened}</div>
            </div>
            <div className="bg-panel border border-edge rounded-lg p-4 border-b-2 border-b-accept">
              <div className="text-xs text-muted mb-1">Accepted</div>
              <div className="text-3xl font-display text-accept">{results.accept}</div>
            </div>
            <div className="bg-panel border border-edge rounded-lg p-4 border-b-2 border-b-reject">
              <div className="text-xs text-muted mb-1">Rejected</div>
              <div className="text-3xl font-display text-reject">{results.reject}</div>
            </div>
            <div className="bg-panel border border-edge rounded-lg p-4">
              <div className="text-xs text-muted mb-1">Hours Saved</div>
              <div className="text-3xl font-display text-accent">{results.hours_saved}h</div>
            </div>
          </div>
          
          <div className="bg-panel border border-edge rounded-xl shadow-lg overflow-hidden">
            <div className="overflow-auto max-h-[400px]">
              <table className="w-full text-left border-collapse text-sm whitespace-nowrap">
                <thead className="bg-[#0f172a] sticky top-0 z-10 border-b border-edge">
                  <tr>
                    <th className="p-3 font-semibold text-gray-300">Part ID</th>
                    <th className="p-3 font-semibold text-gray-300 text-right">0h</th>
                    <th className="p-3 font-semibold text-gray-300 text-right">24h</th>
                    <th className="p-3 font-semibold text-gray-300">Disposition</th>
                    <th className="p-3 font-semibold text-gray-300">Reason</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge">
                  {results.records.map((r, i) => (
                    <tr key={i} className="hover:bg-white/5 transition-colors">
                      <td className="p-3 font-medium text-gray-200">{r.part_id}</td>
                      <td className="p-3 text-right text-gray-400">{r.value_0h?.toFixed(2)}</td>
                      <td className="p-3 text-right text-gray-200">{r.value_24h?.toFixed(2)}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded text-[10px] font-bold ${
                          r.final_decision === 'REJECT' ? 'bg-reject/20 text-reject' :
                          r.final_decision === 'REVIEW' ? 'bg-review/20 text-review' :
                          'bg-accept/10 text-accept'
                        }`}>
                          {r.final_decision}
                        </span>
                      </td>
                      <td className="p-3 text-xs text-muted truncate max-w-xs" title={r.primary_reason}>{r.primary_reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
