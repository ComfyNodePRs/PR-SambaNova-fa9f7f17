import os
import sys

# Add the SambaNova directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .nodes.SambaNova import SambaNovaLLMNode

NODE_CLASS_MAPPINGS = {
    "SambaNovaLLMNode": SambaNovaLLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SambaNovaLLMNode": "SambaNova LLM"
}