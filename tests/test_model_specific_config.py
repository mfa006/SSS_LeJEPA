from PIL import Image

from sss_probe.config import SSSDataConfig
from sss_probe.dataset import SSSDataset
from scripts.run_dense_linear_probe import build_data_config


def test_benthic_settings_are_separate_from_generic_data_settings():
    config = {
        "data": {
            "input_channels": 3,
            "preprocessing": "imagenet",
            "enable_contrast_stretch": True,
        },
        "model": {"name": "benthic_dino"},
        "benthic_dino": {
            "input_channels": 1,
            "preprocessing": "benthic",
            "enable_contrast_stretch": False,
            "mean": 0.77,
            "std": 0.15,
            "feature_variant": "fused",
            "load_tile_labels": False,
        },
    }

    data_config = build_data_config(config)

    assert data_config.input_channels == 1
    assert data_config.preprocessing == "benthic"
    assert data_config.enable_contrast_stretch is False
    assert data_config.benthic_mean == 0.77
    assert data_config.benthic_std == 0.15
    assert data_config.benthic_feature_variant == "fused"
    assert data_config.load_tile_labels is False


def test_non_benthic_model_uses_generic_data_settings():
    config = {
        "data": {
            "input_channels": 3,
            "preprocessing": "imagenet",
            "enable_contrast_stretch": True,
        },
        "model": {"name": "convnext_tiny"},
        "benthic_dino": {
            "input_channels": 1,
            "preprocessing": "benthic",
            "enable_contrast_stretch": False,
        },
    }

    data_config = build_data_config(config)

    assert data_config.input_channels == 3
    assert data_config.preprocessing == "imagenet"
    assert data_config.enable_contrast_stretch is True


def test_dataset_skips_unused_tile_csv_when_disabled(tmp_path):
    image_path = tmp_path / "tile.tiff"
    mask_path = tmp_path / "tile_mask.tiff"
    labels_path = tmp_path / "malformed_labels.csv"
    Image.new("L", (4, 4), color=128).save(image_path)
    Image.new("L", (4, 4), color=1).save(mask_path)
    labels_path.write_text("not,a,valid,label,csv\n")

    dataset = SSSDataset(
        [str(image_path)],
        [str(mask_path)],
        [str(labels_path)],
        config=SSSDataConfig(
            target_size=(4, 4),
            input_channels=1,
            preprocessing="benthic",
            enable_contrast_stretch=False,
            load_tile_labels=False,
        ),
    )

    sample = dataset[0]

    assert sample["labels"].tolist() == [0.0, 0.0, 0.0, 0.0]
