import unittest

from tracks import TRACK_BUILDERS, build_track


class TrackCatalogTests(unittest.TestCase):
    def test_required_tracks_exist(self):
        self.assertIn("rectangle", TRACK_BUILDERS)
        self.assertIn("chicane", TRACK_BUILDERS)

    def test_track_builds_segments(self):
        for name in TRACK_BUILDERS:
            track = build_track(name)
            self.assertGreater(len(track.wall_segments), 0)


if __name__ == "__main__":
    unittest.main()
