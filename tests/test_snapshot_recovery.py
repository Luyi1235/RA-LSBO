import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.recover_compact_qualification import materialize


class SnapshotRecoveryTests(unittest.TestCase):
    def snapshot(self, root):
        snapshot = root/"snapshot"
        snapshot.mkdir()
        original = root/"original"
        index = {}
        for relative in ("lsbo/space.py", "problems/x57/worker.py"):
            content = ("# "+relative+"\n").encode()
            digest = hashlib.sha256(content).hexdigest()
            index[str(original/relative)] = digest
            (snapshot/(digest+".source")).write_bytes(content)
        (snapshot/"index.json").write_text(json.dumps(index))
        return snapshot, original, index

    def test_recovers_original_bytes_into_new_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, original, index = self.snapshot(root)
            destination = root/"isolated"
            result = materialize(snapshot, original, destination)
            self.assertEqual(len(result), 2)
            self.assertFalse(original.exists())
            for filename, expected in result.items():
                self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                materialize(snapshot, original, destination)

    def test_corrupt_blob_is_rejected_before_creating_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, original, index = self.snapshot(root)
            (snapshot/(next(iter(index.values()))+".source")).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                materialize(snapshot, original, root/"isolated")
            self.assertFalse((root/"isolated").exists())

    def test_unrelated_snapshot_paths_are_not_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, original, index = self.snapshot(root)
            index[str(root/"unrelated/secret.py")] = "not-a-valid-blob"
            index[str(original/"runs/old.json")] = "not-a-valid-blob"
            (snapshot/"index.json").write_text(json.dumps(index))
            result = materialize(snapshot, original, root/"isolated")
            self.assertEqual(len(result), 2)
            self.assertFalse((root/"isolated/runs").exists())


if __name__ == "__main__":
    unittest.main()
