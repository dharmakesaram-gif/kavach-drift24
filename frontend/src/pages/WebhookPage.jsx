import React, { useState } from 'react'

export default function WebhookPage() {
  const [eventType, setEventType] = useState('LOT_MAVERICK_ALERT')
  const [targetUrl, setTargetUrl] = useState('https://mock-mes.space-electronics.internal/api/v1/events')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)

  const sendWebhook = async () => {
    setLoading(true)
    setResponse(null)
    try {
      const res = await fetch('/api/v1/webhooks/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_url: targetUrl,
          event_type: eventType,
          lot_id: 'LOT-DEMO-' + Math.floor(Math.random()*1000),
          rejection_rate_pct: 6.8
        })
      })
      const data = await res.json()
      setResponse({ status: res.status, data })
    } catch (err) {
      setResponse({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h2 className="text-2xl font-display font-bold text-white">🔌 MES & ERP Integrations</h2>
        <p className="text-muted text-sm mt-1">Configure and dispatch authenticated webhooks to Manufacturing Execution Systems.</p>
      </div>

      <div className="bg-panel border border-edge rounded-xl p-8 shadow-xl">
        <h3 className="text-lg font-display font-semibold text-white mb-4">Webhook Dispatch Tester</h3>
        
        <div className="space-y-4 max-w-2xl">
          <div>
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Event Type</label>
            <select 
              value={eventType} 
              onChange={e => setEventType(e.target.value)}
              className="bg-[#0f172a] border border-edge text-white rounded-md px-3 py-2 w-full focus:ring-1 focus:ring-accent outline-none"
            >
              <option value="LOT_MAVERICK_ALERT">LOT_MAVERICK_ALERT (&gt;5% Rejection)</option>
              <option value="EARLY_REJECTION_24H">EARLY_REJECTION_24H</option>
              <option value="CHAMBER_THERMAL_TRIP">CHAMBER_THERMAL_TRIP (Safety Override)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Target MES URL</label>
            <input 
              type="text" 
              value={targetUrl} 
              onChange={e => setTargetUrl(e.target.value)}
              className="bg-[#0f172a] border border-edge text-white rounded-md px-3 py-2 w-full focus:ring-1 focus:ring-accent outline-none"
            />
          </div>

          <button 
            onClick={sendWebhook}
            disabled={loading}
            className={`w-full py-3 rounded-lg font-bold tracking-wide transition-all ${
              loading 
                ? 'bg-accent/50 text-ink cursor-not-allowed' 
                : 'bg-accent text-ink hover:bg-accent/90 hover:scale-105'
            }`}
          >
            {loading ? 'DISPATCHING...' : 'DISPATCH TEST PAYLOAD'}
          </button>
        </div>

        {response && (
          <div className="mt-8 animate-in fade-in">
            <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              <span className={response.error ? 'text-reject' : 'text-accept'}>●</span> 
              Dispatch Result
            </h4>
            <div className="bg-ink border border-edge p-4 rounded-lg overflow-auto">
              <pre className="text-xs text-gray-300 font-mono">
                {JSON.stringify(response, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
