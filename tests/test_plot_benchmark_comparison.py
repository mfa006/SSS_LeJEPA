from scripts.plot_benchmark_comparison import (
    CLASS_NAMES,
    metric_value,
    model_label,
    plot_class_precision_and_recall,
)


def test_macro_f1_reads_legacy_m_dice_column():
    row = {"test_mDice_pct": "94.40"}

    assert metric_value(row, "test_macro_f1_pct") == 94.4


def test_model_label_includes_parameter_count_for_legacy_rows():
    row = {"model": "resnet34", "feature_dim": "512"}

    assert model_label(row) == "ResNet-34 (D=512, 21.28M params)"


def test_class_precision_and_recall_are_saved_as_separate_figures(tmp_path):
    row = {"model": "resnet34", "feature_dim": "512"}
    for metric in ("precision", "recall"):
        for class_name in CLASS_NAMES:
            row[f"test_{metric}_{class_name.lower()}_pct"] = "95.0"

    old_combined_plot = tmp_path / "benchmark_class_precision_recall.png"
    old_combined_plot.write_bytes(b"old combined plot")

    plot_class_precision_and_recall([row], tmp_path)

    assert (tmp_path / "benchmark_class_precision.png").is_file()
    assert (tmp_path / "benchmark_class_recall.png").is_file()
    assert not old_combined_plot.exists()
