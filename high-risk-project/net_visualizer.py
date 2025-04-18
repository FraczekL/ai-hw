import torch
from torchview import draw_graph
# Make sure Detector is imported from your models file
from high_risk_project import OurBrainReadingCNNTransformerHybridWithResNet
import os
import graphviz # Import graphviz if you need the set_jupyter_format line

# --- Optional: Workaround for PATH issues ---
# Keep this if needed
# graphviz_bin_path = 'C:\\Program Files\\Graphviz\\bin' # Example for Windows
# if os.path.exists(graphviz_bin_path):
#     os.environ["PATH"] += os.pathsep + graphviz_bin_path
# -------------------------------------------

# --- Optional: Fix for VS Code rendering format if needed ---
# Keep this if needed
# graphviz.set_jupyter_format('png')
# -----------------------------------------------------------


# Use the Detector model
model = OurBrainReadingCNNTransformerHybridWithResNet()
# Use a sample input size appropriate for your model
input_data = torch.zeros(32, 3, 224, 224)

dummy_input = {
    "track_left": torch.zeros(1, 10, 2),
    "track_right": torch.zeros(1, 10, 2),
}

# Generate the graph object WITHOUT the graph_attr argument
model_graph = draw_graph(
    model=model,
    input_data=input_data,
    device='meta',
    expand_nested=True,
    graph_name='CNN_Hybrid' # Optional: Change graph name
    # REMOVED graph_attr from here
)

# --- Set graph attributes AFTER creating the graph object ---
# Access the underlying graphviz object (usually via .visual_graph)
# and set the rankdir attribute directly.
model_graph.visual_graph.graph_attr.update({
    'rankdir': 'TB', # LR = Left to Right
    'dpi': '300'
})
# ----------------------------------------------------------

try:
    # Explicitly save the graph to a PNG file
    output_filename = f'{model.__class__.__name__}_graph_vertical'
    model_graph.visual_graph.render(filename=output_filename, format='png', cleanup=True)
    print(f"Diagram saved as {output_filename}.png")
except Exception as e:
    print(f"Error during rendering or saving: {e}")