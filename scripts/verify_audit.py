import sys
import traceback
from pathlib import Path

# Add src to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

print("Checking imports...")
try:
    from scenarioops.observability import observe, get_logger
    print(" [x] scenarioops.observability")
    
    from scenarioops.llm.transport import RequestsTransport
    print(" [x] scenarioops.llm.transport")
    
    from scenarioops.graph.nodes.charter import run_charter_node
    print(" [x] scenarioops.graph.nodes.charter")
    
    from scenarioops.ui.utils import ROOT, SRC_DIR
    print(" [x] scenarioops.ui.utils")
    
    from scenarioops.ui.styles import PIPELINE_CSS
    print(" [x] scenarioops.ui.styles")
    
    from scenarioops.ui.components.pipeline import render_pipeline_boxes
    print(" [x] scenarioops.ui.components.pipeline")

    print("\nAll imports successful.")
except Exception as e:
    print(f"\n[!] Verification FAILED: {e}")
    with open("verify_error.log", "w") as f:
        traceback.print_exc(file=f)
    sys.exit(1)
