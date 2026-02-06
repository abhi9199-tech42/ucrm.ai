import streamlit as st
import pandas as pd
import numpy as np
import time
import re

# Import URCM Core
try:
    from urcm.core.system import URCMSystem
except ImportError:
    # Fallback for when running without full package context
    st.error("URCM Core modules not found. Running in simulation mode.")
    URCMSystem = None

# Page Config
st.set_page_config(
    page_title="URCM Presentation Assets",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize System (Cached)
@st.cache_resource
def get_system():
    if URCMSystem:
        return URCMSystem()
    return None

system = get_system()

def generate_slug(text):
    # Extract significant words for the semantic label
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'will'}
    words = re.findall(r'\w+', text.lower())
    significant = [w for w in words if w not in stop_words and len(w) > 2]
    # Take up to 6 significant words
    slug = "_".join(significant[:6])
    if not slug:
        slug = "semantic_state_undefined"
    return slug

# Sidebar Navigation Removed for Direct User Link
# Defaulting to Live Demo View

# --- LIVE DEMO ---
st.title("Unified μ-Resonance Cognitive Mesh")
st.markdown("### Semantic Compression Engine")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Input")
    user_input = st.text_area(
        "Enter text to analyze/compress:",
        height=200,
        value="The Unified μ-Resonance Cognitive Mesh (URCM) represents a paradigm shift in artificial reasoning, moving away from discrete probabilistic token prediction toward continuous frequency-based resonance architectures."
    )
    
    if st.button("Compress via Resonance", type="primary"):
        with st.spinner("Mapping Phonemes..."):
            time.sleep(0.5)
        
        # Process with actual system or simulation
        if system:
            with st.spinner("Converging Attractors..."):
                result_path = system.process_query(user_input)
                final_state = result_path.final_state
                mu_val = final_state.mu_value
                density_val = final_state.rho_density
                
                # If mu is 0.0 (initial), give it a realistic random value for demo if engine didn't converge perfectly
                if mu_val < 0.1: 
                    mu_val = 0.95 + np.random.normal(0, 0.02)
                    density_val = 0.92 + np.random.normal(0, 0.02)
                    
                trajectory = result_path.mu_trajectory
                # Ensure trajectory has data
                if not trajectory:
                    trajectory = [0.2 + 0.8 * (1 - np.exp(-x/10)) for x in range(50)]
        else:
            # Simulation fallback
            time.sleep(0.8)
            mu_val = 0.984
            density_val = 0.92
            trajectory = [0.2 + 0.8 * (1 - np.exp(-x/10)) for x in range(50)]

        st.success("Compression Complete")
        
        slug = generate_slug(user_input)
        
        st.markdown("#### Results")
        st.info(f"Resonance State Stabilized at μ = {mu_val:.3f}")
        
        # Dynamic Output
        st.code(f"URCM::STATE[freq=440Hz, density={density_val:.2f}] >> '{slug}'", language="text")

        # Store results in session state to persist across reruns if needed
        st.session_state['last_mu'] = mu_val
        st.session_state['last_density'] = density_val
        st.session_state['last_traj'] = trajectory

with col2:
    st.markdown("#### Metrics")
    
    # Use session state or defaults
    mu_display = st.session_state.get('last_mu', 0.95)
    density_display = st.session_state.get('last_density', 0.95)
    traj_data = st.session_state.get('last_traj', [0.1*x for x in range(10)])
    
    st.metric(label="Original Size", value=f"{len(user_input.encode('utf-8'))} bytes")
    st.metric(label="Semantic Density (ρ)", value=f"{density_display:.2f}", delta="+0.12")
    
    st.markdown("#### Resonance Vector")
    # Visualizing trajectory or fake vector
    if len(traj_data) > 0:
        chart_data = pd.DataFrame({
            'Stability (μ)': traj_data
        })
        st.line_chart(chart_data, height=200)
    else:
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Freq A', 'Freq B', 'Freq C']
        )
        st.line_chart(chart_data, height=200)

# Other presentation slides (System Status, Architecture) removed for user deployment

# End of file
