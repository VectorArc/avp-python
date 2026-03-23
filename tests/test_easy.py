"""Tests for the avp easy API: think(), generate()."""

from avp.easy import clear_cache


class TestEdgeCases:
    def test_clear_cache(self):
        # Should not raise even when caches are empty
        clear_cache()
