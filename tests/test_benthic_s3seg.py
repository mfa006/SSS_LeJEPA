import json
from types import SimpleNamespace

import numpy as np
import torch

from sss_probe.benthic_s3seg import (
    _downsample_masks,
    _evaluate_split,
    _extract_tile_samples,
    _fit_linear_probes,
    run_benthic_s3seg_benchmark,
)
from sss_probe.metrics import compute_segmentation_metrics


def test_downsample_masks_matches_benthic_s3seg_center_sampling():
    mask = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)

    downsampled = _downsample_masks(mask, 2, 2)

    np.testing.assert_array_equal(downsampled, [[[5, 7], [13, 15]]])


def test_extract_tile_samples_runs_batched_inference_deterministically():
    class TinyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {
                "image": torch.full((1, 4, 4), float(index)),
                "mask": torch.arange(16, dtype=torch.long).reshape(4, 4) % 4,
            }

    batch_sizes = []

    def fake_inference(
        _model,
        images,
        device,
        normalize,
        return_tensors,
        include_embeddings,
        feature_indices,
    ):
        assert device == torch.device("cpu")
        assert normalize is False
        assert return_tensors is True
        assert include_embeddings is False
        assert feature_indices == (4,)
        batch_sizes.append(len(images))
        values = images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        stages = [
            values.expand(len(images), 2, 2, 2)
            for _ in range(5)
        ]
        return None, None, [stages[index] for index in feature_indices]

    features, labels = _extract_tile_samples(
        model=None,
        dataset=TinyDataset(),
        inference_fn=fake_inference,
        device=torch.device("cpu"),
        batch_size=2,
        pixels_per_tile=2,
        seed=7,
        stage_indices=(4,),
    )
    second_features, second_labels = _extract_tile_samples(
        model=None,
        dataset=TinyDataset(),
        inference_fn=fake_inference,
        device=torch.device("cpu"),
        batch_size=2,
        pixels_per_tile=2,
        seed=7,
        stage_indices=(4,),
    )

    assert batch_sizes == [2, 1, 2, 1]
    assert set(features) == set(labels) == {4}
    assert features[4].shape == (3, 2, 2)
    assert labels[4].shape == (3, 2)
    np.testing.assert_array_equal(features[4], second_features[4])
    np.testing.assert_array_equal(labels[4], second_labels[4])


def test_fit_linear_probes_trains_only_requested_fused_stage_on_device():
    rng = np.random.default_rng(42)
    features = {4: rng.normal(size=(20, 4, 3)).astype(np.float16)}
    labels = {
        4: np.tile(np.array([0, 1, 0, 1], dtype=np.uint8), (20, 1))
    }
    val_features = {4: rng.normal(size=(8, 4, 3)).astype(np.float16)}
    val_labels = {
        4: np.tile(np.array([0, 1, 0, 1], dtype=np.uint8), (8, 1))
    }

    probes, histories = _fit_linear_probes(
        feature_samples=features,
        label_samples=labels,
        val_feature_samples=val_features,
        val_label_samples=val_labels,
        n_train_tiles=20,
        few_shot=(1.0,),
        seed=42,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lr_scheduler="cosine",
        early_stopping_patience=2,
    )

    assert set(probes) == {(4, 1.0)}
    assert set(histories) == {(4, 1.0)}
    assert histories[(4, 1.0)]["best_epoch"] in (1, 2)
    assert len(histories[(4, 1.0)]["train_loss"]) == 2
    logits = probes[(4, 1.0)](
        torch.from_numpy(features[4].reshape(-1, 3).astype(np.float32))
    )
    assert logits.shape == (80, 4)


def test_evaluate_split_reports_one_fused_linear_probe():
    class TinyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "image": torch.full((1, 4, 4), float(index)),
                "mask": torch.arange(16, dtype=torch.long).reshape(4, 4) % 2,
            }

    class ConstantClassifier(torch.nn.Module):
        def forward(self, features):
            logits = features.new_zeros((len(features), 4))
            logits[:, 0] = 1.0
            return logits

    def fake_inference(
        _model,
        images,
        device,
        normalize,
        return_tensors,
        include_embeddings,
        feature_indices,
    ):
        assert return_tensors is True
        assert feature_indices == (4,)
        features = torch.zeros((len(images), 3, 2, 2), dtype=torch.float32)
        return None, None, [features]

    def fake_report(y_true, _y_pred, labels):
        class_values = np.asarray(labels)
        counts = np.asarray([(y_true == label).sum() for label in class_values])
        zeros = np.zeros(len(labels))
        return "report", (None, zeros, zeros, zeros, zeros, counts)

    records = _evaluate_split(
        model=None,
        dataset=TinyDataset(),
        split_name="test",
        classifiers={(4, 1.0): ConstantClassifier()},
        inference_fn=fake_inference,
        report_fn=fake_report,
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        stage_indices=(4,),
    )

    assert len(records) == 1
    assert records[0]["stage"] == "stage4_fused"
    assert records[0]["method"] == "torch_linear"
    assert records[0]["few_shot"] == 1.0
    assert "mDice" in records[0]
    assert "precision_class_0" in records[0]


def test_benthic_benchmark_writes_standard_artifacts(tmp_path, monkeypatch):
    class TinyDataset:
        config = SimpleNamespace(benthic_mean=0.5, benthic_std=0.25)

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "image": torch.full((1, 4, 4), index / 4, dtype=torch.float32),
                "mask": torch.arange(16, dtype=torch.long).reshape(4, 4) % 4,
                "filename": f"tile_{index}.tiff",
            }

    def fake_inference(
        _model,
        images,
        device,
        normalize,
        return_tensors,
        include_embeddings,
        feature_indices,
    ):
        assert normalize is False
        assert return_tensors is True
        assert include_embeddings is False
        assert feature_indices == (4,)
        images = images.to(device)
        pattern = torch.arange(16, device=device, dtype=images.dtype).reshape(1, 1, 4, 4)
        pattern = pattern.expand(len(images), -1, -1, -1) / 16
        features = torch.cat((images, images + pattern, pattern), dim=1)
        return None, None, [features]

    def fake_report(y_true, y_pred, labels):
        metrics = compute_segmentation_metrics(
            np.asarray(y_pred), np.asarray(y_true), num_classes=len(labels)
        )
        values = (
            None,
            np.asarray([metrics[f"precision_class_{i}"] for i in labels]),
            np.asarray([metrics[f"recall_class_{i}"] for i in labels]),
            np.asarray([metrics[f"macro_f1"] for _ in labels]),
            np.asarray([metrics[f"iou_class_{i}"] for i in labels]),
            np.asarray([metrics[f"support_class_{i}"] for i in labels]),
        )
        return "test report", values

    monkeypatch.setattr(
        "sss_probe.benthic_s3seg._benthic_utils",
        lambda: SimpleNamespace(
            run_inference=fake_inference,
            generate_evaluation_report=fake_report,
        ),
    )
    result = run_benthic_s3seg_benchmark(
        model=torch.nn.Identity(),
        train_dataset=TinyDataset(),
        val_dataset=TinyDataset(),
        test_dataset=TinyDataset(),
        device="cpu",
        results_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        config_file="test.yaml",
        batch_size=2,
        pixels_per_tile=4,
        few_shot=(1.0,),
        epochs=1,
        classifier_batch_size=8,
        lr_scheduler="none",
        early_stopping_patience=1,
    )

    result_dir = tmp_path / "results"
    assert result["test_metrics"]["split"] == "test"
    assert (result_dir / "benthic_dino_linear_probe.json").is_file()
    assert (result_dir / "benthic_dino_metrics.json").is_file()
    assert (result_dir / "benthic_dino_training_curves.png").is_file()
    assert (result_dir / "benthic_dino_test_predictions.png").is_file()
    assert (tmp_path / "checkpoints" / "benthic_dino_linear_probe.pt").is_file()
    with (result_dir / "benthic_dino_metrics.json").open() as metrics_file:
        metrics = json.load(metrics_file)
    assert metrics["best_epoch"] == 1
    assert metrics["test_metrics"]["method"] == "torch_linear"
