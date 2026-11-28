"""
Tests for the image corrupters in robosuite/utils/observables.py (occlusion, salt and pepper, JPEG).

Each corrupter is checked on its own on synthetic images, and then once through an Observable to make sure it fits
the sensor -> corrupter -> filter pipeline that camera observables use.
"""
import numpy as np
import pytest

from robosuite.utils.observables import (
    Observable,
    create_jpeg_corrupter,
    create_occlusion_corrupter,
    create_salt_and_pepper_corrupter,
    sensor,
)


def _gradient_image(h=64, w=96):
    """Smooth rgb test image with values well inside (0, 255), so corrupted pixels are easy to tell apart."""
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.stack([50 + xs, 60 + ys, 100 + (xs + ys) // 2], axis=-1)
    return img.astype(np.uint8)


def test_occlusion_patch_size_and_fill():
    np.random.seed(0)
    img = _gradient_image()
    corrupter = create_occlusion_corrupter(min_size=(0.25, 0.5), fill=0)
    for _ in range(20):
        out = corrupter(img)
        assert out.shape == img.shape and out.dtype == img.dtype
        covered = np.all(out == 0, axis=-1)
        # exactly one 16 x 48 rectangle is filled, everything else is untouched
        assert covered.sum() == 16 * 48
        rows, cols = np.where(covered.any(axis=1))[0], np.where(covered.any(axis=0))[0]
        assert rows[-1] - rows[0] + 1 == 16 and cols[-1] - cols[0] + 1 == 48
        assert np.array_equal(out[~covered], img[~covered])


def test_occlusion_does_not_modify_input_and_moves():
    np.random.seed(1)
    img = _gradient_image()
    ref = img.copy()
    corrupter = create_occlusion_corrupter(min_size=0.1, max_size=0.4, num_patches=2, fill=(255, 0, 255))
    outs = [corrupter(img) for _ in range(5)]
    assert np.array_equal(img, ref)
    assert any(not np.array_equal(outs[0], o) for o in outs[1:])
    changed = np.any(outs[0] != img, axis=-1)
    assert np.all(outs[0][changed] == (255, 0, 255))


def test_occlusion_depth_and_full_image():
    np.random.seed(2)
    depth = np.random.random_sample((32, 32, 1)).astype(np.float32) + 0.5
    out = create_occlusion_corrupter(min_size=1.0, fill=0.0)(depth)
    assert out.shape == depth.shape and out.dtype == depth.dtype
    assert np.all(out == 0.0)


def test_occlusion_rejects_bad_sizes():
    with pytest.raises(AssertionError):
        create_occlusion_corrupter(min_size=0.0)
    with pytest.raises(AssertionError):
        create_occlusion_corrupter(min_size=0.5, max_size=0.2)
    with pytest.raises(AssertionError):
        create_occlusion_corrupter(min_size=0.5, max_size=1.5)


def test_salt_and_pepper_fraction():
    np.random.seed(3)
    img = _gradient_image(200, 200)
    out = create_salt_and_pepper_corrupter(amount=0.1, salt_vs_pepper=0.25)(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    salt = np.all(out == 255, axis=-1)
    pepper = np.all(out == 0, axis=-1)
    # 40000 pixels: expect about 1000 salt and 3000 pepper
    assert abs(salt.mean() - 0.025) < 0.005
    assert abs(pepper.mean() - 0.075) < 0.005
    clean = ~(salt | pepper)
    assert np.array_equal(out[clean], img[clean])


def test_salt_and_pepper_extremes():
    np.random.seed(4)
    img = _gradient_image()
    assert np.array_equal(create_salt_and_pepper_corrupter(amount=0.0)(img), img)
    assert np.all(create_salt_and_pepper_corrupter(amount=1.0, salt_vs_pepper=1.0)(img) == 255)
    assert np.all(create_salt_and_pepper_corrupter(amount=1.0, salt_vs_pepper=0.0)(img) == 0)
    depth = np.ones((16, 16, 1), dtype=np.float32)
    out = create_salt_and_pepper_corrupter(amount=1.0, salt_vs_pepper=0.0, low=0.0, high=1.0)(depth)
    assert out.dtype == np.float32 and np.all(out == 0.0)


def test_jpeg_error_grows_as_quality_drops():
    img = _gradient_image()
    # add texture so that compression has something to throw away
    np.random.seed(5)
    img = np.clip(img.astype(int) + np.random.randint(-30, 30, img.shape), 0, 255).astype(np.uint8)
    errs = []
    for quality in (95, 50, 5):
        out = create_jpeg_corrupter(quality)(img)
        assert out.shape == img.shape and out.dtype == np.uint8
        errs.append(np.abs(out.astype(float) - img.astype(float)).mean())
    assert 0 < errs[0] < errs[1] < errs[2]


def test_jpeg_keeps_channel_order():
    # a pure red image must come back red, not blue
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 200
    out = create_jpeg_corrupter(90)(img)
    assert out[..., 0].mean() > 180 and out[..., 2].mean() < 20


def test_jpeg_grayscale_and_bad_input():
    np.random.seed(6)
    gray = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
    out = create_jpeg_corrupter(10, max_quality=30)(gray)
    assert out.shape == gray.shape and out.dtype == np.uint8
    with pytest.raises(AssertionError):
        create_jpeg_corrupter(50)(gray.astype(np.float32))
    with pytest.raises(AssertionError):
        create_jpeg_corrupter(0)
    with pytest.raises(AssertionError):
        create_jpeg_corrupter(80, max_quality=40)


def test_corrupters_in_observable():
    np.random.seed(7)
    img = _gradient_image()

    @sensor(modality="image")
    def camera(obs_cache):
        return img

    occlude = create_occlusion_corrupter(min_size=0.25, fill=0)
    jpeg = create_jpeg_corrupter(40)
    obs = Observable(name="cam_image", sensor=camera, corrupter=lambda x: jpeg(occlude(x)), sampling_rate=20)
    obs.update(timestep=0.05, obs_cache={}, force=True)
    assert obs.obs.shape == img.shape and obs.obs.dtype == np.uint8
    assert not np.array_equal(obs.obs, img)
    # the occluded block survives compression as a dark region
    assert (obs.obs.max(axis=-1) < 30).sum() > 0.8 * 16 * 24
