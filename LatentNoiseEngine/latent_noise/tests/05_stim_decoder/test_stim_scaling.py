import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from latent_noise.engine.pipeline import LatentNoisePipeline
from latent_noise.decoder.matching import get_rotated_surface_code_hx, get_logical_z_op

def test_66_syndrome_correctness():
    H = get_rotated_surface_code_hx(3)
    assert H.shape == (4, 9)
    assert np.all(H.sum(axis=1) >= 2)

def test_68_logical_error_detection():
    d = 3
    Lz = get_logical_z_op(d)
    errors = Lz.reshape(1, -1)
    syndromes = (errors @ get_rotated_surface_code_hx(d).T) % 2
    from latent_noise.stim_layer.sampler import decode_and_check_logical_scaling
    from latent_noise.decoder.matching import build_decoder_for_distance
    matching = build_decoder_for_distance(d)
    logical_errors = decode_and_check_logical_scaling(d, errors, syndromes, matching)
    assert logical_errors[0] == 1

def test_70_distance_scaling():
    cfg = {
        "steps": 10, "seed": 42, "shots": 5000, 
        "sigma_temporal": 0.02, "target_hazard": 0.01
    }
    h3 = np.mean(LatentNoisePipeline({**cfg, "d": 3}).run()["hazards"])
    h5 = np.mean(LatentNoisePipeline({**cfg, "d": 5}).run()["hazards"])
    h7 = np.mean(LatentNoisePipeline({**cfg, "d": 7}).run()["hazards"])
    assert h3 > h5 >= h7

def test_92_distance_ordering():
    # Use higher noise to ensure non-zero counts
    cfg = {"steps": 10, "seed": 1, "shots": 5000, "sigma_temporal": 0.05}
    h3 = np.mean(LatentNoisePipeline({**cfg, "d": 3}).run()["hazards"])
    h5 = np.mean(LatentNoisePipeline({**cfg, "d": 5}).run()["hazards"])
    assert h3 > h5
