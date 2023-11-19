from huggingface_hub import login,HfApi
import time  
ticks = time.time()
login("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")
api = HfApi()

final_submit = f"final_submit_v3_{int(ticks)}"
api.create_repo(f"xxyyy123/{final_submit}")
api.upload_folder(
    folder_path="final_v3_test",
    repo_id=f"xxyyy123/{final_submit}",
    repo_type="model",
)