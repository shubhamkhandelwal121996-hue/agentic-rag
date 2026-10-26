from pathlib import Path
from agentic_rag.core.Agentic_rag1 import build_graph

# Build the graph
graph = build_graph()

# Generate its diagram
image = graph.get_graph().draw_mermaid_png()

# Save beside draw_graph.py
output_path = Path(__file__).parent / "rag_graph.png"
output_path.write_bytes(image)

print(f"Graph saved: {output_path}")