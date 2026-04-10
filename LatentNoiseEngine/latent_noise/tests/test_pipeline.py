import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.pipeline import LatentNoisePipeline

def test_pipeline_runs():
    pipe = LatentNoisePipeline({"steps": 10})
    out = pipe.run()
    assert len(out["hazards"]) == 10