import unittest
import numpy as np
from backend.audio.features import AcousticFeatureExtractor, get_mel_filterbank


class TestAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = AcousticFeatureExtractor(n_mfcc=13, target_sr=16000)

    def test_feature_names_length(self):
        names = self.extractor.get_feature_names()
        # 13 MFCC + rms + zcr + centroid + bandwidth + rolloff + flux + dom_freq = 20
        self.assertEqual(len(names), 20)
        self.assertIn("mfcc_1", names)
        self.assertIn("spectral_centroid", names)
        self.assertIn("dominant_frequency", names)

    def test_filterbank_shape(self):
        fb = get_mel_filterbank(num_filters=26, n_fft=1024, sample_rate=16000)
        self.assertEqual(fb.shape, (26, 513))
        # Non-negative
        self.assertTrue(np.all(fb >= 0))

    def test_extract_window_features(self):
        sr = 16000
        # 1.0s sine wave at 440 Hz
        t = np.linspace(0, 1.0, sr, endpoint=False)
        window = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        feat_dict, vec, mag = self.extractor.extract_window_features(window)
        self.assertEqual(len(vec), 20)
        self.assertAlmostEqual(feat_dict["rms"], 0.5 / np.sqrt(2), delta=0.05)
        # Dominant frequency should be close to 440 Hz
        self.assertAlmostEqual(feat_dict["dominant_frequency"], 440.0, delta=20.0)


if __name__ == "__main__":
    unittest.main()
