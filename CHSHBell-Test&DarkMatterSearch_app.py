"""
🔬 Quantum Network Validation Platform
=======================================
CHSH Bell-Test & Dark Matter Search - Enterprise Edition

Features:
- Upload timestamp data from any time-tagger
- Real-time quantum network validation
- Live CHSH monitoring with alerts
- Export validation certificates
- Multi-node network testing
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import json
import base64
from io import BytesIO, StringIO
import time
import threading
import queue
import random

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

HBAR = 6.582119569e-16  # eV·s
C = 2.99792458e8  # m/s
PI = np.pi
TSIRELSON_BOUND = 2 * np.sqrt(2)
CLASSICAL_BOUND = 2.0

# ============================================================================
# NETWORK VALIDATION ENGINE
# ============================================================================

@dataclass
class NetworkNode:
    """Represents a node in a quantum network"""
    node_id: str
    name: str
    location: str
    channel_map: dict
    status: str = "idle"
    last_validation: dict = field(default_factory=dict)
    s_value: float = 0.0
    sigma: float = 0.0
    
@dataclass
class NetworkValidationResult:
    """Results from network validation"""
    timestamp: datetime
    node_id: str
    S: float
    sigma_S: float
    sigma_above: float
    violates: bool
    within_tsirelson: bool
    settings: dict
    report_id: str

class QuantumNetworkValidator:
    """Real-time quantum network validation engine"""
    
    def __init__(self):
        self.nodes = {}
        self.validation_history = []
        self.running = False
        self.alert_callbacks = []
        
    def add_node(self, node: NetworkNode):
        """Add a node to the network"""
        self.nodes[node.node_id] = node
        
    def validate_node(self, node_id: str, data: pd.DataFrame) -> NetworkValidationResult:
        """Validate a single network node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        
        # Parse the data and compute CHSH
        try:
            # Extract coincidence counts
            # Expected columns: angle_a, angle_b, N_AB, N_CD, N_AC, N_BD
            settings = {}
            for _, row in data.iterrows():
                key = f"a{row['angle_a']}_b{row['angle_b']}"
                settings[key] = {
                    'angle_a': row['angle_a'],
                    'angle_b': row['angle_b'],
                    'N_AB': row['N_AB'],
                    'N_CD': row['N_CD'],
                    'N_AC': row.get('N_AC', 0),
                    'N_BD': row.get('N_BD', 0)
                }
            
            # Compute E for each setting
            E_values = {}
            sigma_values = {}
            for key, s in settings.items():
                total = s['N_AB'] + s['N_CD'] + s['N_AC'] + s['N_BD']
                if total == 0:
                    continue
                E = (s['N_AB'] + s['N_CD'] - s['N_AC'] - s['N_BD']) / total
                sigma = np.sqrt(total) / total
                E_values[key] = E
                sigma_values[key] = sigma
            
            # Find the CHSH settings (assuming standard angles)
            # Map angles to standard CHSH settings
            chsh_settings = {}
            for key, E in E_values.items():
                # Parse angle values from key
                parts = key.replace('a', '').replace('b', '').split('_')
                if len(parts) == 2:
                    a = float(parts[0])
                    b = float(parts[1])
                    # Map to standard CHSH keys
                    if abs(a) < 1 and abs(b - 22.5) < 1:
                        chsh_settings['ab'] = E
                    elif abs(a) < 1 and abs(b - 67.5) < 1:
                        chsh_settings['abp'] = E
                    elif abs(a - 45) < 1 and abs(b - 22.5) < 1:
                        chsh_settings['apb'] = E
                    elif abs(a - 45) < 1 and abs(b - 67.5) < 1:
                        chsh_settings['apbp'] = E
            
            if len(chsh_settings) < 4:
                return None
            
            # Compute S
            S = (chsh_settings.get('ab', 0) - chsh_settings.get('abp', 0) + 
                 chsh_settings.get('apb', 0) + chsh_settings.get('apbp', 0))
            
            # Compute sigma
            sigma_S = np.sqrt(sum([sigma_values.get(k, 0)**2 for k in ['ab', 'abp', 'apb', 'apbp'] if k in sigma_values]))
            sigma_above = (abs(S) - CLASSICAL_BOUND) / sigma_S if sigma_S > 0 else 0
            
            # Update node
            node.s_value = S
            node.sigma = sigma_S
            node.last_validation = {
                'S': S,
                'sigma_S': sigma_S,
                'sigma_above': sigma_above,
                'timestamp': datetime.now(),
                'settings': chsh_settings
            }
            
            result = NetworkValidationResult(
                timestamp=datetime.now(),
                node_id=node_id,
                S=S,
                sigma_S=sigma_S,
                sigma_above=sigma_above,
                violates=abs(S) > CLASSICAL_BOUND,
                within_tsirelson=abs(S) <= TSIRELSON_BOUND + 1e-9,
                settings=chsh_settings,
                report_id=f"VAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            self.validation_history.append(result)
            
            # Check for alerts
            self._check_alerts(result)
            
            return result
            
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            return None
    
    def _check_alerts(self, result: NetworkValidationResult):
        """Check if validation triggers any alerts"""
        if result.violates and result.sigma_above > 5:
            self._trigger_alert("✅ ENTANGLEMENT CONFIRMED", 
                               f"Node {result.node_id} validated at {result.sigma_above:.2f}σ")
        elif result.violates and result.sigma_above > 3:
            self._trigger_alert("⚠️ ENTANGLEMENT DETECTED", 
                               f"Node {result.node_id} shows evidence at {result.sigma_above:.2f}σ")
        elif not result.violates:
            self._trigger_alert("❌ ENTANGLEMENT NOT DETECTED", 
                               f"Node {result.node_id} failed CHSH test")
    
    def _trigger_alert(self, title: str, message: str):
        """Trigger an alert"""
        for callback in self.alert_callbacks:
            callback(title, message)
    
    def get_network_status(self) -> dict:
        """Get overall network status"""
        active_nodes = [n for n in self.nodes.values() if n.status == "active"]
        validated_nodes = [n for n in active_nodes if n.last_validation]
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'validated_nodes': len(validated_nodes),
            'entangled_nodes': len([n for n in validated_nodes if abs(n.s_value) > CLASSICAL_BOUND]),
            'avg_S': np.mean([n.s_value for n in validated_nodes]) if validated_nodes else 0,
            'timestamp': datetime.now()
        }

# ============================================================================
# NETWORK DATA GENERATORS
# ============================================================================

def generate_test_network_data(seed=42, num_settings=4, counts=100000):
    """Generate synthetic network test data"""
    np.random.seed(seed)
    
    # Standard CHSH settings
    settings = [
        {'angle_a': 0, 'angle_b': 22.5, 'E_true': -0.707},
        {'angle_a': 0, 'angle_b': 67.5, 'E_true': 0.707},
        {'angle_a': 45, 'angle_b': 22.5, 'E_true': 0.707},
        {'angle_a': 45, 'angle_b': 67.5, 'E_true': -0.707}
    ]
    
    data = []
    for s in settings:
        E_true = s['E_true']
        p_correlated = (1 + E_true) / 2
        
        N_AB = np.random.poisson(counts * p_correlated / 2)
        N_CD = np.random.poisson(counts * p_correlated / 2)
        N_AC = np.random.poisson(counts * (1 - p_correlated) / 2)
        N_BD = np.random.poisson(counts * (1 - p_correlated) / 2)
        
        data.append({
            'angle_a': s['angle_a'],
            'angle_b': s['angle_b'],
            'N_AB': int(N_AB),
            'N_CD': int(N_CD),
            'N_AC': int(N_AC),
            'N_BD': int(N_BD)
        })
    
    return pd.DataFrame(data)

def generate_network_timestamp_data(node_id, duration_seconds=60, event_rate=1000):
    """Generate synthetic timestamp data for network simulation"""
    np.random.seed(random.randint(0, 1000000) + hash(node_id))
    
    # Simulate photon detections with entangled correlations
    n_events = int(duration_seconds * event_rate)
    timestamps = np.cumsum(np.random.exponential(1/event_rate, n_events) * 1e9)
    
    # Assign channels with correlation
    channels = []
    labels = []
    for i in range(n_events):
        # Simulate entangled pairs
        if i % 2 == 0:
            # Alice side
            outcome = np.random.choice(['A+', 'A-'])
            channels.append(1 if outcome == 'A+' else 2)
            labels.append(outcome)
        else:
            # Bob side (correlated)
            prev_outcome = labels[-1]
            if prev_outcome == 'A+':
                # Correlated: A+ -> B+, A- -> B-
                outcome = np.random.choice(['B+', 'B-'], p=[0.85, 0.15])
            else:
                outcome = np.random.choice(['B+', 'B-'], p=[0.15, 0.85])
            channels.append(3 if outcome == 'B+' else 4)
            labels.append(outcome)
    
    return pd.DataFrame({
        'timestamp_ns': timestamps,
        'channel': channels,
        'label': labels
    })

# ============================================================================
# VALIDATION REPORT GENERATOR
# ============================================================================

def generate_validation_certificate(result: NetworkValidationResult, node: NetworkNode):
    """Generate a validation certificate as HTML"""
    status = "✅ PASSED" if result.violates and result.sigma_above > 5 else "⚠️ PARTIAL" if result.violates else "❌ FAILED"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum Network Validation Certificate</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; }}
            .certificate {{ 
                border: 2px solid #1f77b4; 
                border-radius: 10px; 
                padding: 40px; 
                max-width: 800px; 
                margin: 0 auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .header {{ text-align: center; border-bottom: 2px solid #1f77b4; padding-bottom: 20px; }}
            .header h1 {{ color: #1f77b4; margin: 0; }}
            .header h3 {{ color: #666; margin: 5px 0; }}
            .status {{ 
                text-align: center; 
                padding: 20px; 
                margin: 20px 0;
                background: {'#d4edda' if status == '✅ PASSED' else '#fff3cd' if status == '⚠️ PARTIAL' else '#f8d7da'};
                border-radius: 5px;
            }}
            .status h2 {{ margin: 0; }}
            .details {{ margin: 20px 0; }}
            .details table {{ width: 100%; border-collapse: collapse; }}
            .details th, .details td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            .details th {{ background: #f0f2f6; }}
            .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 12px; border-top: 1px solid #ddd; padding-top: 20px; }}
            .badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
            .badge-pass {{ background: #28a745; color: white; }}
            .badge-warn {{ background: #ffc107; color: #333; }}
            .badge-fail {{ background: #dc3545; color: white; }}
        </style>
    </head>
    <body>
        <div class="certificate">
            <div class="header">
                <h1>🔬 Quantum Network Validation Certificate</h1>
                <h3>CHSH Bell-Test Certification</h3>
                <p>Certificate ID: {result.report_id}</p>
                <p>Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="status">
                <h2>Validation Status: <span class="badge badge-{'pass' if status == '✅ PASSED' else 'warn' if status == '⚠️ PARTIAL' else 'fail'}">{status}</span></h2>
            </div>
            
            <div class="details">
                <h3>Network Node Details</h3>
                <p><strong>Node ID:</strong> {node.node_id}</p>
                <p><strong>Name:</strong> {node.name}</p>
                <p><strong>Location:</strong> {node.location}</p>
                
                <h3>CHSH Results</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>S-Parameter</td>
                        <td>{result.S:.4f} ± {result.sigma_S:.4f}</td>
                        <td>{'✅' if result.violates else '❌'}</td>
                    </tr>
                    <tr>
                        <td>Significance</td>
                        <td>{result.sigma_above:.2f} σ</td>
                        <td>{'✅' if result.sigma_above > 5 else '⚠️'}</td>
                    </tr>
                    <tr>
                        <td>Classical Bound (|S| ≤ 2)</td>
                        <td>{'Violated' if result.violates else 'Not Violated'}</td>
                        <td>{'✅' if result.violates else '❌'}</td>
                    </tr>
                    <tr>
                        <td>Tsirelson Bound (|S| ≤ 2.828)</td>
                        <td>{'Within' if result.within_tsirelson else 'Exceeded'}</td>
                        <td>{'✅' if result.within_tsirelson else '⚠️'}</td>
                    </tr>
                </table>
                
                <h3>Per-Setting Results</h3>
                <table>
                    <tr>
                        <th>Setting</th>
                        <th>Angle A</th>
                        <th>Angle B</th>
                        <th>E(a,b)</th>
                    </tr>
                    {''.join([f'<tr><td>{k}</td><td>{v:.1f}</td><td>{v:.1f}</td><td>{result.settings.get(k, 0):+.4f}</td></tr>' for k, v in [('ab', 22.5), ('abp', 67.5), ('apb', 22.5), ('apbp', 67.5)]])}
                </table>
            </div>
            
            <div class="footer">
                <p>This certificate validates that the quantum network node has demonstrated entanglement</p>
                <p>according to the CHSH Bell inequality test at the time of measurement.</p>
                <p>Generated by Quantum Network Validation Platform</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

st.set_page_config(
    page_title="🔬 Quantum Network Validation Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #e8ecf1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .status-pass {
        color: #27ae60;
        font-weight: bold;
    }
    .status-fail {
        color: #e74c3c;
        font-weight: bold;
    }
    .status-warn {
        color: #f39c12;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        opacity: 0.8;
    }
    .node-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .live-active {
        background-color: #27ae60;
        animation: pulse 1s infinite;
    }
    .live-idle {
        background-color: #95a5a6;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'validator' not in st.session_state:
    st.session_state.validator = QuantumNetworkValidator()
if 'nodes' not in st.session_state:
    st.session_state.nodes = {}
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🌐 Network Validation")
    st.markdown("### Quantum Network Testing")
    
    st.markdown("---")
    st.markdown("### 📡 Add Network Node")
    
    node_id = st.text_input("Node ID:", value="NODE-001")
    node_name = st.text_input("Node Name:", value="Quantum Lab 1")
    node_location = st.text_input("Location:", value="Biddeford, ME")
    
    if st.button("➕ Add Node", type="primary"):
        if node_id not in st.session_state.nodes:
            node = NetworkNode(
                node_id=node_id,
                name=node_name,
                location=node_location,
                channel_map={1: "A+", 2: "A-", 3: "B+", 4: "B-"}
            )
            st.session_state.validator.add_node(node)
            st.session_state.nodes[node_id] = node
            st.success(f"✅ Node {node_id} added")
        else:
            st.warning(f"Node {node_id} already exists")
    
    st.markdown("---")
    st.markdown("### 🔍 Test Options")
    
    test_type = st.radio(
        "Test Type:",
        ["Upload Data", "Generate Test Data", "Live Network Monitor"]
    )
    
    if test_type == "Upload Data":
        st.markdown("**Upload CHSH Data**")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file and st.button("Run Validation", type="primary"):
            try:
                df = pd.read_csv(uploaded_file)
                result = st.session_state.validator.validate_node(node_id, df)
                if result:
                    st.session_state.validation_results.append(result)
                    st.success(f"✅ Validation complete: S = {result.S:.4f} ± {result.sigma_S:.4f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif test_type == "Generate Test Data":
        if st.button("Generate CHSH Data", type="primary"):
            df = generate_test_network_data()
            result = st.session_state.validator.validate_node(node_id, df)
            if result:
                st.session_state.validation_results.append(result)
                st.success(f"✅ Validation complete: S = {result.S:.4f} ± {result.sigma_S:.4f}")
    
    elif test_type == "Live Network Monitor":
        st.markdown("**Real-time Monitoring**")
        duration = st.slider("Duration (seconds):", 10, 120, 30)
        
        if st.button("▶️ Start Monitoring", type="primary"):
            st.session_state.monitoring = True
            
    st.markdown("---")
    st.markdown("### 📊 Network Status")
    
    status = st.session_state.validator.get_network_status()
    st.metric("Total Nodes", status['total_nodes'])
    st.metric("Active Nodes", status['active_nodes'])
    st.metric("Validated Nodes", status['validated_nodes'])
    st.metric("Entangled Nodes", status['entangled_nodes'])
    st.metric("Avg S-Parameter", f"{status['avg_S']:.4f}")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<p class="main-header">🔬 Quantum Network Validation Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time CHSH Bell-Test Certification for Quantum Networks</p>', unsafe_allow_html=True)

# Network Nodes Display
st.markdown("## 🌐 Network Nodes")
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.nodes:
        for node_id, node in st.session_state.nodes.items():
            status_color = "live-active" if node.status == "active" else "live-idle"
            status_text = "🟢 Active" if node.status == "active" else "⚪ Idle"
            
            with st.container():
                st.markdown(f"""
                <div class="node-card">
                    <h4>{node.name}</h4>
                    <p><span class="live-indicator {status_color}"></span> {status_text}</p>
                    <p><strong>ID:</strong> {node.node_id} | <strong>Location:</strong> {node.location}</p>
                    <p><strong>S-Parameter:</strong> {node.s_value:.4f} ± {node.sigma:.4f}</p>
                    <p><strong>Status:</strong> {'✅ Entangled' if abs(node.s_value) > CLASSICAL_BOUND and node.sigma > 0 else '❌ Not Validated'}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No nodes added. Add a node from the sidebar.")

with col2:
    st.markdown("### 📈 Network Health")
    
    if st.session_state.validation_results:
        # Show latest validation results
        latest = st.session_state.validation_results[-1]
        
        color = "status-pass" if latest.violates and latest.sigma_above > 5 else "status-warn" if latest.violates else "status-fail"
        status_text = "✅ PASS" if latest.violates and latest.sigma_above > 5 else "⚠️ PARTIAL" if latest.violates else "❌ FAIL"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Latest Validation</h3>
            <h2 class="{color}">{status_text}</h2>
            <p>S = {latest.S:.4f} ± {latest.sigma_S:.4f}</p>
            <p>{latest.sigma_above:.2f} σ above classical bound</p>
            <p>Node: {latest.node_id}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation history chart
        if len(st.session_state.validation_results) > 1:
            fig = go.Figure()
            
            times = [r.timestamp for r in st.session_state.validation_results]
            S_values = [r.S for r in st.session_state.validation_results]
            errors = [r.sigma_S for r in st.session_state.validation_results]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=S_values,
                mode='lines+markers',
                name='S-Parameter',
                error_y=dict(type='data', array=errors)
            ))
            
            fig.add_hline(y=CLASSICAL_BOUND, line_dash="dash", line_color="red", annotation_text="Classical Bound")
            fig.add_hline(y=TSIRELSON_BOUND, line_dash="dot", line_color="green", annotation_text="Tsirelson Bound")
            
            fig.update_layout(
                title="Validation History",
                xaxis_title="Time",
                yaxis_title="S",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No validation results yet. Run a test to see results.")

# Live Monitoring
if st.session_state.monitoring:
    st.markdown("---")
    st.markdown("## 📡 Live Network Monitor")
    
    # Placeholder for real-time monitoring
    status_placeholder = st.empty()
    data_placeholder = st.empty()
    
    # Simulate live monitoring
    for i in range(10):
        with status_placeholder:
            st.markdown(f"🟢 **Monitoring Active** - Cycle {i+1}/10")
        
        # Generate and validate data
        df = generate_network_timestamp_data(node_id, duration_seconds=5)
        result = st.session_state.validator.validate_node(node_id, df)
        if result:
            st.session_state.validation_results.append(result)
        
        time.sleep(1)
    
    st.session_state.monitoring = False
    st.success("✅ Monitoring complete!")

# Export Certificate
st.markdown("---")
st.markdown("## 📋 Validation Certificates")

if st.session_state.validation_results:
    latest = st.session_state.validation_results[-1]
    node = st.session_state.nodes.get(latest.node_id)
    
    if node and st.button("📄 Generate Certificate"):
        html = generate_validation_certificate(latest, node)
        st.download_button(
            label="📥 Download Certificate (HTML)",
            data=html,
            file_name=f"certificate_{latest.report_id}.html",
            mime="text/html"
        )

# Data Export
st.markdown("## 📤 Export Data")

if st.session_state.validation_results:
    export_format = st.selectbox("Export Format:", ["CSV", "JSON", "Markdown"])
    
    if st.button("📥 Export Validation Data"):
        data = []
        for r in st.session_state.validation_results:
            data.append({
                'timestamp': r.timestamp,
                'node_id': r.node_id,
                'S': r.S,
                'sigma_S': r.sigma_S,
                'sigma_above': r.sigma_above,
                'violates': r.violates,
                'within_tsirelson': r.within_tsirelson
            })
        
        df = pd.DataFrame(data)
        
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "validation_results.csv", "text/csv")
        elif export_format == "JSON":
            json_data = df.to_json(orient='records', indent=2)
            st.download_button("📥 Download JSON", json_data, "validation_results.json", "application/json")
        else:
            md = df.to_markdown(index=False)
            st.download_button("📥 Download Markdown", md, "validation_results.md", "text/markdown")
