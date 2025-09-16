from datasets import load_dataset

splits = ['jl', 'cq', 'yt', 'sh', 'hz']
type_dataset = 'delivery_'
for s in splits:
    ds = load_dataset("Cainiao-AI/LaDe-D", split=f"{type_dataset}{s}")
    ds.to_csv(f"{type_dataset}{s}.csv")
