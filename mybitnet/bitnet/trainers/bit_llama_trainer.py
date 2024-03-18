from transformers import Trainer
from huggingface_hub import Repository
import os
import shutil
from pathlib import Path
from typing import Optional

class BitLlamaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        # Copy modeling_bit_llama.py to output_dir
        if self.is_world_process_zero():
            modeling_file = "modeling_bit_llama.py"
            modeling_path = os.path.join(os.path.dirname(__file__), "..", "models", "bit_llama", modeling_file)
            output_modeling_path = os.path.join(self.args.output_dir, modeling_file)
            shutil.copy(modeling_path, output_modeling_path)

            # Add modeling_bit_llama.py to .gitattributes
            with open(os.path.join(self.args.output_dir, ".gitattributes"), "a") as f:
                f.write(f"{modeling_file} filter=lfs diff=lfs merge=lfs -text\n")

        # Call the parent method to push the model, tokenizer, and modeling_bit_llama.py
        result = super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

        return result