from datasets import load_dataset

splits = ['jl', 'cq', 'yt', 'sh', 'hz']
for s in splits:
    ds = load_dataset("Cainiao-AI/LaDe-D", split=f"delivery_{s}")
    ds.to_csv(f"delivery_{s}.csv")

    ds = load_dataset("Cainiao-AI/LaDe-P", split=f"pickup_{s}")
    ds.to_csv(f"pickup_{s}.csv")
