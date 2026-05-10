import { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, TrendingUp, Users, ChevronRight, X, Phone, Mail, Activity, DollarSign } from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8001/api';

function App() {
  const [kpis, setKpis] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCustomer, setSelectedCustomer] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [kpiRes, custRes] = await Promise.all([
          axios.get(`${API_BASE}/kpis`),
          axios.get(`${API_BASE}/customers`)
        ]);
        setKpis(kpiRes.data);
        setCustomers(custRes.data);
      } catch (err) {
        console.error("Error fetching data", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const getRiskColor = (prob) => {
    if (prob > 0.7) return 'var(--risk-high)';
    if (prob > 0.4) return 'var(--risk-med)';
    return 'var(--risk-low)';
  };

  const getRiskBg = (prob) => {
    if (prob > 0.7) return 'var(--risk-high-bg)';
    if (prob > 0.4) return 'var(--risk-med-bg)';
    return 'var(--risk-low-bg)';
  };

  const getRiskLabel = (prob) => {
    if (prob > 0.7) return 'High';
    if (prob > 0.4) return 'Med';
    return 'Low';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo-section">
          <Activity className="logo-icon" size={24} color="var(--accent-color)" />
          <h1>AMC Retain (Ensemble: CatBoost + WTTE)</h1>
        </div>
        <div className="user-profile">
          <span>Sarah J. (Sales Lead)</span>
          <div className="avatar">SJ</div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* KPIs */}
        <section className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Revenue at Risk (30d)</span>
              <DollarSign size={18} className="text-red" />
            </div>
            <div className="kpi-value">{kpis ? kpis.revenue_at_risk : '...'}</div>
          </div>
          
          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Avg Portfolio Risk</span>
              <TrendingUp size={18} className="text-blue" />
            </div>
            <div className="kpi-value">{kpis ? kpis.avg_churn_risk : '...'}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-title">Expiring Contracts</span>
              <Users size={18} className="text-green" />
            </div>
            <div className="kpi-value">{kpis ? kpis.expiring_30_days : '...'}</div>
          </div>
        </section>

        <div className="layout-grid">
          {/* Action Queue */}
          <section className="queue-section">
            <div className="section-header">
              <h2>Prioritized Action Queue</h2>
              <span className="subtitle">Sorted by Churn Risk × Revenue</span>
            </div>
            
            {loading ? (
              <div className="loading">Loading ML predictions...</div>
            ) : (
              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Account</th>
                      <th>Risk Level</th>
                      <th>Risk Score</th>
                      <th>Monthly Value</th>
                      <th>Priority Score</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {customers.map((c, i) => (
                      <tr 
                        key={c.id} 
                        className={`table-row ${selectedCustomer?.id === c.id ? 'selected' : ''}`}
                        onClick={() => setSelectedCustomer(c)}
                      >
                        <td className="font-medium">{c.id}</td>
                        <td>
                          <span 
                            className="badge" 
                            style={{ 
                              backgroundColor: getRiskBg(c.risk_score), 
                              color: getRiskColor(c.risk_score),
                              border: `1px solid ${getRiskColor(c.risk_score)}`
                            }}
                          >
                            {getRiskLabel(c.risk_score)}
                          </span>
                        </td>
                        <td>{(c.risk_score * 100).toFixed(1)}%</td>
                        <td>${c.monthly_charges.toFixed(2)}</td>
                        <td><span className="font-mono">{c.priority_score.toFixed(1)}</span></td>
                        <td><ChevronRight size={16} className="chevron" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* Details Sidebar */}
          {selectedCustomer ? (
            <aside className="detail-sidebar">
              <button className="close-btn" onClick={() => setSelectedCustomer(null)}>
                <X size={18} />
              </button>
              
              <div className="customer-header">
                <h2>{selectedCustomer.id}</h2>
                <div className="risk-huge" style={{ color: getRiskColor(selectedCustomer.risk_score) }}>
                  {(selectedCustomer.risk_score * 100).toFixed(1)}% Risk
                </div>
                <div style={{ marginTop: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  <strong>RNN Expected Lifespan:</strong> ~{selectedCustomer.expected_days} days
                </div>
              </div>

              <div className="sidebar-section">
                <h3><ShieldAlert size={16} /> AI Risk Drivers</h3>
                <p className="insight-sub">Key factors predicting churn:</p>
                <div className="drivers-list">
                  {selectedCustomer.drivers.map((d, i) => (
                    <div key={i} className="driver-item">
                      <div className="driver-dot"></div>
                      <span>{d}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="sidebar-section">
                <h3>Playbook Action</h3>
                <div className="action-box">
                  <p>{selectedCustomer.recommended_action}</p>
                </div>
              </div>

              <div className="action-buttons">
                <button className="btn btn-primary"><Phone size={14}/> Log Call</button>
                <button className="btn btn-secondary"><Mail size={14}/> Email Quote</button>
              </div>
            </aside>
          ) : (
            <aside className="detail-sidebar-empty">
              <Activity size={32} className="empty-icon" />
              <p>Select an account from the queue to view AI insights and retention playbook.</p>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
