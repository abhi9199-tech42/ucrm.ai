import streamlit as st

st.set_page_config(page_title="Use Cases & Approach", layout="wide")

st.title("Use Cases & Builder Approach")

st.markdown("""
### 🚀 How to Use URCM

The Unified μ-Resonance Cognitive Mesh is designed for high-density semantic processing. Here is how you can leverage it:

*   **Semantic Compression:** Reduce large text corpora into stable resonance states (μ-states) without losing semantic density.
*   **Cognitive Search:** Use frequency-based vectors to find "truth" matches rather than just keyword overlaps.
*   **Low-Latency Reasoning:** Replace heavy LLM inference with O(1) attractor network lookups for common reasoning patterns.
""")

st.markdown("---")

st.markdown("""
### 🛠️ Builder's Approach

If you are a developer or system architect building on top of URCM, follow this integration pattern:

1.  **Initialize the Mesh:** Load the core `URCMSystem` with your domain-specific phoneme mappings.
2.  **Map Inputs:** Convert raw user input into frequency vectors.
3.  **Converge:** Allow the system to settle into a stable attractor state.
4.  **Extract:** Read the final `μ` score and semantic density `ρ` to determine confidence.

```python
# Example Integration Pattern
from urcm.core import URCMSystem

system = URCMSystem()
state = system.process("User query input")

if state.stability_index > 0.95:
    execute_high_confidence_action(state.payload)
else:
    fallback_to_traditional_search()
```
""")

st.markdown("---")

st.info("""
### 🤝 Connect for Product Integration

**Are you building a specific product?**

If you need product-specific customization or have architectural questions, please connect with us directly via our repository.

**👉 [Leave a comment on our GitHub Discussions/Issues](https://github.com/abhi9199-tech42/ucrm.ai)**
""")
