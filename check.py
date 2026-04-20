import pandas as pd

report = pd.read_csv("results/evaluation_report.csv")
valid = report[report["eval_correct"].notna()].copy()
valid["eval_correct"] = valid["eval_correct"].astype(bool)

# 整体准确率
overall = valid["eval_correct"].mean() * 100
print(f"Overall Accuracy: {overall:.1f}% ({int(valid['eval_correct'].sum())}/{len(valid)})")

# 按 prompt mode
print("\n=== By Prompt Mode ===")
by_mode = (
    valid.groupby("prompt_mode")["eval_correct"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "Accuracy", "sum": "Correct", "count": "Total"})
    .sort_values("Accuracy", ascending=False)
)
by_mode["Accuracy"] = (by_mode["Accuracy"] * 100).round(1).astype(str) + "%"
print(by_mode.to_string())

# 付费模型名字
print("\n=== Model IDs ===")
print(report["model_id"].unique())