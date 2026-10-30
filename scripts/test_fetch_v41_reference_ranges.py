import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fetch_v41_reference_ranges as fetcher

class Response(io.BytesIO):
    def __init__(self, payload, status=206, content_range="bytes 3-5/10"):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}

class FetchTest(unittest.TestCase):
    def test_valid_range_is_verified_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = hashlib.sha256(b"abc").hexdigest()
            with patch.object(fetcher.urllib.request, "urlopen", return_value=Response(b"abc")) as request:
                fetcher.fetch(root, "x", 3, 5, digest)
                self.assertEqual((root / "x").read_bytes(), b"abc")
                fetcher.fetch(root, "x", 3, 5, digest)
                self.assertEqual(request.call_count, 1)

    def test_bad_responses_never_replace_existing_file(self):
        for response in [Response(b"abc", status=200), Response(b"ab"), Response(b"abcd"), Response(b"xyz")]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "x").write_bytes(b"old")
                with patch.object(fetcher.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises(RuntimeError):
                        fetcher.fetch(root, "x", 3, 5, hashlib.sha256(b"abc").hexdigest())
                self.assertEqual((root / "x").read_bytes(), b"old")
                self.assertEqual(sorted(p.name for p in root.iterdir()), ["x"])

if __name__ == "__main__":
    unittest.main()
