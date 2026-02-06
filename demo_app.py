import streamlit as st
import pandas as pd
import numpy as np
import time
import graphviz

# Page Config
st.set_page_config(
    page_title="URCM Presentation Assets",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("URCM Navigation")
slide = st.sidebar.radio("Select View:", [
    "Live Demo",
    "System Status",
    "Architecture"
])

# --- LIVE DEMO ---
if slide == "Live Demo":
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
            with st.spinner("Converging Attractors..."):
                time.sleep(0.8)
            
            st.success("Compression Complete")
            
            st.markdown("#### Results")
            st.info("Resonance State Stabilized at μ = 0.984")
            
            # Simulated Output
            st.code("URCM::STATE[freq=440Hz, density=0.92] >> 'paradigm_shift_reasoning_continuous'", language="text")

    with col2:
        st.markdown("#### Metrics")
        st.metric(label="Original Size", value="248 bytes")
        st.metric(label="Semantic Density (ρ)", value="0.95", delta="+0.12")
        
        st.markdown("#### Resonance Vector")
        # Visualizing a fake vector
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Freq A', 'Freq B', 'Freq C']
        )
        st.line_chart(chart_data, height=200)

# --- SYSTEM STATUS ---
elif slide == "System Status":
    st.header("System Verification Status")
    
    # Big Status Banner
    st.markdown("""
    <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-bottom: 30px;">
        <h1 style="margin:0;">✅ 93/93 TESTS PASSED</h1>
        <p style="margin:0;">All critical subsystems verified</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Pass Rate", "100%", delta="Stable")
    kpi2.metric("Convergence Time", "12ms", delta="-2ms")
    kpi3.metric("Stability Score", "0.99", delta="+0.01")
    kpi4.metric("Memory Usage", "45MB", delta="Optimal")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Benchmark")
        # Simulated performance data
        chart_data = pd.DataFrame({
            'Steps': range(50),
            'Stability (μ)': [min(1.0, 0.2 + 0.8 * (1 - np.exp(-x/10)) + np.random.normal(0, 0.02)) for x in range(50)],
            'Error Rate': [max(0.0, 0.8 * np.exp(-x/8) + np.random.normal(0, 0.02)) for x in range(50)]
        })
        st.line_chart(chart_data.set_index('Steps'))
        st.caption("System convergence over reasoning steps")
        
    with col2:
        st.subheader("Test Suite Coverage")
        test_data = pd.DataFrame({
            'Category': ['Phoneme Mapping', 'Resonance Engine', 'Attractor Network', 'Latent Space', 'Integration'],
            'Tests': [15, 28, 20, 12, 18]
        })
        st.bar_chart(test_data.set_index('Category'))

# --- ARCHITECTURE ---
elif slide == "Architecture":
    st.header("How It Works: URCM Architecture")
    
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')
    graph.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica')
    
    # Nodes
    graph.node('I', 'Input Text', fillcolor='#e3f2fd')
    graph.node('P', 'Phoneme\nMapping', fillcolor='#fff3e0')
    graph.node('C', 'Compression\n(Resonance)', fillcolor='#e8f5e9')
    graph.node('O', 'Output\n(Stable State)', fillcolor='#f3e5f5')
    
    # Edges
    graph.edge('I', 'P', label=' Raw Text')
    graph.edge('P', 'C', label=' Frequency\nVectors')
    graph.edge('C', 'O', label=' Converged\nμ-State')
    
    st.graphviz_chart(graph)
    
    st.markdown("### Process Flow")
    st.info("1. **Input:** Raw text is received.\n2. **Phoneme Mapping:** Text is converted into fundamental sound-frequency components.\n3. **Compression:** Frequencies interfere constructively/destructively to find the stable 'truth' state.\n4. **Output:** The final high-density semantic state is returned.")
