import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face.crop import crop_face, expanded_bbox  # noqa: E402
from src.geo import analyze_scene  # noqa: E402
from src.search.osint import contextual_queries, split_handle  # noqa: E402


def test_expanded_bbox_is_clipped_to_image():
    assert expanded_bbox((10, 10, 30, 30), (32, 32, 3), 0.35) == (3, 3, 32, 32)


def test_crop_face_returns_expanded_region():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_face(image, (40, 40, 60, 60), 0.35).shape[:2] == (34, 34)


def test_contextual_queries_split_handle_and_keep_public_platforms():
    assert split_handle("supreme__Sahil") == "supreme Sahil"
    queries = contextual_queries("GourishJulka", "web3 speaker")
    assert '"Gourish Julka" web3 speaker' in queries
    assert any("linkedin.com/posts" in query for query in queries)


def test_scene_analysis_reports_observable_cues():
    result = analyze_scene(np.full((20, 40, 3), 200, dtype=np.uint8))
    assert result["lighting"] == "high"
    assert result["composition"] == "landscape"