import unittest
import pyiron_dpd


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = pyiron_dpd.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
