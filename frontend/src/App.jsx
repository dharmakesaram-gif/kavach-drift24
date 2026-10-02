import { useState, useEffect } from 'react'
import { Activity, LayoutDashboard, BarChart2, AlertTriangle, Search, SlidersHorizontal, ShieldCheck, Cpu } from 'lucide-react'
import OverviewPage from './pages/OverviewPage'
import DistributionPage from './pages/DistributionPage'
import FlaggedPage from './pages/FlaggedPage'
import InspectorPage from './pages/InspectorPage'
import WhatIfPage from './pages/WhatIfPage'
import ATEIngestionPage from './pages/ATEIngestionPage'
import ChamberLivePage from './pages/ChamberLivePage'
import CoCPage from './pages/CoCPage'
import WebhookPage from './pages/WebhookPage'
import SPRTPage from './pages/SPRTPage'
import BenchmarksPage from './pages/BenchmarksPage'
import ModelCardPage from './pages/ModelCardPage'

function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch data from FastAPI backend
    fetch('/api/v1/dashboard/data')
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch dashboard data")
        return res.json()
      })
      .then(json => {
        setData(json.data)
        setLoading(false)
      })
      .catch(err => {
        console.error(err)
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const navItems = [
    { id: 'overview', label: 'Executive Screening Overview', icon: <LayoutDashboard size={18} /> },
    { id: 'distribution', label: 'Lot Distribution & Dynamic PAT', icon: <BarChart2 size={18} /> },
    { id: 'flagged', label: 'Flagged Parts & Disposition', icon: <AlertTriangle size={18} /> },
    { id: 'inspector', label: 'QA Inspector Deep-Dive', icon: <Search size={18} /> },
    { id: 'sprt', label: 'SPRT Sequential Decision', icon: <Activity size={18} /> },
    { id: 'benchmarks', label: 'Benchmarks & Baselines', icon: <BarChart2 size={18} /> },
    { id: 'whatif', label: 'Interactive What-If Simulation', icon: <SlidersHorizontal size={18} /> },
    { id: 'ate', label: 'Raw ATE Ingestion', icon: <Activity size={18} /> },
    { id: 'chamber', label: 'Live Chamber Telemetry', icon: <Activity size={18} /> },
    { id: 'coc', label: 'Compliance & CoC', icon: <ShieldCheck size={18} /> },
    { id: 'webhooks', label: 'MES Integrations', icon: <Activity size={18} /> },
    { id: 'modelcard', label: 'Model Card & Limitations', icon: <ShieldCheck size={18} /> },
  ]

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#060a12] border-r border-edge flex flex-col shrink-0">
        <div className="p-6 border-b border-edge">
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="text-accent" />
            <h1 className="text-lg font-bold text-white tracking-tight">Kavach Drift24</h1>
          </div>
          <p className="text-xs text-muted">SIH26170 | High-Rel ESS</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors ${
                activeTab === item.id 
                  ? 'bg-accent/10 border-l-2 border-accent text-accent font-medium' 
                  : 'text-gray-400 hover:bg-accent/5 hover:text-gray-200 border-l-2 border-transparent'
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-edge">
          <div className="text-xs text-muted mb-2 font-medium uppercase tracking-wider">Live Integrations</div>
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <div className="w-2 h-2 rounded-full bg-accept animate-pulse"></div>
            REST API (Port 8000)
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-300 mt-2">
            <ShieldCheck size={14} className="text-accept" />
            Chamber Interlock ARMED
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-transparent relative">
        <div className="max-w-[1400px] mx-auto p-8">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-[60vh]">
              <Activity className="animate-spin text-accent mb-4" size={32} />
              <p className="text-muted font-display text-lg">Initializing Physics Engine & Models...</p>
            </div>
          ) : error ? (
            <div className="bg-red-900/20 border border-reject/50 p-6 rounded-lg">
              <h2 className="text-reject font-bold text-lg mb-2 flex items-center gap-2">
                <AlertTriangle /> Backend Connection Failed
              </h2>
              <p className="text-gray-300 mb-4">{error}</p>
              <p className="text-sm text-muted">Ensure the FastAPI backend is running on http://localhost:8000 (cd app && uvicorn run_api:app --reload)</p>
            </div>
          ) : (
            <div className="animate-in fade-in duration-500">
              {activeTab === 'overview' && <OverviewPage data={data} />}
              {activeTab === 'distribution' && <DistributionPage data={data} />}
              {activeTab === 'flagged' && <FlaggedPage data={data} />}
              {activeTab === 'inspector' && <InspectorPage data={data} />}
              {activeTab === 'sprt' && <SPRTPage data={data} />}
              {activeTab === 'benchmarks' && <BenchmarksPage data={data} />}
              {activeTab === 'whatif' && <WhatIfPage data={data} />}
              {activeTab === 'ate' && <ATEIngestionPage />}
              {activeTab === 'chamber' && <ChamberLivePage />}
              {activeTab === 'coc' && <CoCPage data={data} />}
              {activeTab === 'webhooks' && <WebhookPage />}
              {activeTab === 'modelcard' && <ModelCardPage data={data} />}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
