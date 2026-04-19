import unittest

from tracks import TRACK_LIBRARY, build_track_definition, track_names


class TrackCatalogTests(unittest.TestCase):
    def test_has_at_least_four_demo_tracks(self):
        self.assertGreaterEqual(len(track_names()), 4)

    def test_required_tracks_exist(self):
        self.assertIn("rectangle", TRACK_LIBRARY)
        self.assertIn("chicane", TRACK_LIBRARY)

    def test_track_definition_includes_spawn_and_heading(self):
        for name in track_names():
            definition = build_track_definition(name)
            self.assertGreater(len(definition.track.wall_segments), 0)
            self.assertEqual(len(definition.spawn), 2)
            self.assertIsInstance(definition.heading_deg, float)


if __name__ == "__main__":
    unittest.main()
