import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from study.acquire import save_json, sha
from study.geometry import exact_taxonomy_permutations
from study.native_cohort import freeze, freeze_layout
from study.native_measure import transfer_metric, vector
from study.native_routes import PROTOCOL, calibrated, read_json, validate_registration
from study.router import RequestError


class NativeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_json(PROTOCOL)

    def test_direct_transport_requires_distinct_inputs_and_stable_route(self):
        first, second = [1, 0], [0, 1]
        stable = [(first, "route-a"), (second, "route-a")] * 2
        self.assertEqual(calibrated(stable, ["route-a"])["transport"], "input_sensitive")
        switched = [*stable[:3], (second, "route-b")]
        self.assertEqual(calibrated(switched, ["route-a", "route-b"])["transport"], "response_alias_changed")
        collapsed = [(first, "route-a")] * 4
        self.assertEqual(calibrated(collapsed, ["route-a"])["transport"], "no_control_separation")

    def test_exact_taxonomy_support_includes_identity_once(self):
        groups = np.array(["a", "b", "b", "a", "b"])
        permutations = exact_taxonomy_permutations(groups)
        self.assertEqual(permutations.shape, (12, 5))
        np.testing.assert_array_equal(permutations[0], np.arange(5))
        np.testing.assert_array_equal(groups[permutations], np.broadcast_to(groups, permutations.shape))
        self.assertEqual(len(np.unique(permutations, axis=0)), 12)

    def test_blocked_audit_exits_before_reading_or_extracting_media(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            routes = path / "routes.json"
            save_json(routes, dict(protocol_sha256=sha(PROTOCOL.read_bytes()), native_acquisition_ready=False))
            with self.assertRaisesRegex(ValueError, "RouterAI route audit"):
                freeze(path / "native", path / "unavailable-manifest.json", routes)
            self.assertFalse((path / "native").exists())

    def test_registry_rejects_missing_family_even_if_other_modalities_exist(self):
        self.protocol["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "distinct development and held-out families"):
            validate_registration(self.protocol)

    def test_audio_vector_path_sends_typed_bytes_and_rejects_a_route_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sound = b"fixture-only-wav-bytes"
            (path / "sound.wav").write_bytes(sound)
            row = {"source_id": "sample", "assets": {"audio": {"path": "sound.wav", "sha256": sha(sound)}}}
            model = dict(modality="audio", id="family", model="test/model",
                         response_model="expected-route", dimension=2)
            with patch("study.native_measure.embedding", return_value=([[1, 2]], "expected-route")) as mocked:
                np.testing.assert_array_equal(vector(path, path, row, model), [1, 2])
                request_input = mocked.call_args.args[2]
                content = request_input[0]["content"]
                self.assertEqual(len(content), 1)
                self.assertEqual(content[0]["type"], "input_audio")
            with patch("study.native_measure.embedding", return_value=([[1, 2]], "another-route")):
                with self.assertRaisesRegex(RequestError, "different model alias"):
                    vector(path, path, row, model)

    def manifest(self):
        datasets = {name: dict(origin=f"https://example.org/{name}", role=name, license="CC0",
                               training_overlap_assessment="synthetic test")
                    for name in ("discovery", "replication")}
        rows = []
        for name in datasets:
            for role in self.protocol["concept_split"]:
                for fold in ("A", "B"):
                    unit = f"{name}-{role}-{fold}"
                    rows.append(dict(source_id=unit, dataset=name, concept=role, group="test",
                                     concept_split=role, fold=fold, recording_sha256=sha(unit.encode()),
                                     license="CC0", source_url="https://example.org/sample", text_authorship="fixture",
                                     dependency_ids=[unit],
                                     assets={m: {"sha256": sha((unit + m).encode())}
                                             for m in ("text", "image", "audio", "video")}))
        return dict(version=1, datasets=datasets, sources=rows)

    def test_independent_panel_layout_checks_hashes_and_source_dependencies(self):
        self.protocol["minimum_concepts_per_panel"] = 1
        self.protocol["minimum_sources_per_concept_fold"] = 1
        original = self.manifest()
        _, rows, panels = freeze_layout(original, self.protocol, set())
        self.assertEqual(len(panels), 4)
        self.assertEqual(len({x["creator_component"] for x in rows}), 8)
        conflicting = copy.deepcopy(original)
        conflicting["sources"][1]["assets"]["video"] = conflicting["sources"][0]["assets"]["video"]
        with self.assertRaisesRegex(ValueError, "shared source component"):
            freeze_layout(conflicting, self.protocol, set())
        exposed = original["sources"][-1]["recording_sha256"]
        with self.assertRaisesRegex(ValueError, "exposed pilot recording"):
            freeze_layout(original, self.protocol, {exposed})

    def test_metric_scale_cannot_refit_to_the_held_out_values(self):
        rdm = np.array([[0., .1, .4], [.1, 0., .2], [.4, .2, 0.]])
        groups = np.array(["group"] * 3)
        dev = (["d1", "d2", "d3"], groups, ([rdm] * 2, [2 * rdm] * 2))
        test = (["t1", "t2", "t3"], groups, ([rdm] * 2, [3 * rdm] * 2))
        measured = transfer_metric(dev, test, ((0, 1), (1, 0)), 12, 13, "fixture")
        self.assertEqual(measured["nonnegative_scales"], [2., 2.])
        self.assertAlmostEqual(measured["relative_rmse"], 1 / 3, places=10)
