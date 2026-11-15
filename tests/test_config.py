import pytest
from pydantic import ValidationError

from miragebench.config import load_arm, load_base, merged_config
from miragebench.config.schema import ArmConfig, DataSpec, SplitConfig


def test_base_loads():
    base = load_base("configs/base.yaml")
    assert base.model.arch == "yolo11s"
    assert base.model.imgsz == 1280
    assert len(base.seeds) >= 1


def test_all_arms_load_and_merge():
    import glob
    base = load_base("configs/base.yaml")
    for p in glob.glob("configs/arms/*.yaml"):
        arm = load_arm(p)
        eff = merged_config(base, arm)
        assert "_arm" in eff
        assert isinstance(eff["_overridden"], list)


def test_dataspec_rejects_incoherent():
    with pytest.raises(ValidationError):
        ArmConfig(name="bad", data=DataSpec(train_source="real", synth_count=100))
    with pytest.raises(ValidationError):
        ArmConfig(name="bad2", data=DataSpec(train_source="synth", real_count=100))


def test_arm_name_no_spaces():
    with pytest.raises(ValidationError):
        ArmConfig(name="bad name", data=DataSpec(train_source="real"))


def test_split_must_sum_to_one():
    with pytest.raises(ValidationError):
        SplitConfig(train=0.5, val=0.3, test=0.3)
    SplitConfig(train=0.8, val=0.15, test=0.05)  # ok


def test_override_unknown_key_rejected():
    base = load_base("configs/base.yaml")
    arm = ArmConfig(name="x", data=DataSpec(train_source="real"), overrides={"nonexistent": 1})
    with pytest.raises(ValueError):
        merged_config(base, arm)
