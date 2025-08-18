import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch


import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from local_apis.fasterwhisperapi import (
    WhisperLitAPI,
    TranscriptionRequest,
    TranscriptionResponse,
)


class TestWhisperLitAPI(unittest.TestCase):

    def setUp(self):

        self.mock_model = MagicMock()
        self.mock_model.transcribe.return_value = (
            [MagicMock(text="test transcription")],
            None,
        )

        self.patcher = patch(
            "local_apis.fasterwhisperapi.WhisperModel", return_value=self.mock_model
        )
        self.mock_whisper_model_class = self.patcher.start()

        self.api = WhisperLitAPI()
        self.api.setup(device="cpu")

    def tearDown(self):
        self.patcher.stop()

    def _create_dummy_audio_request(
        self, sample_rate, num_channels, duration_ms=100, language="vi"
    ):
        """Creates a dummy TranscriptionRequest."""
        samples_per_channel = int(sample_rate * (duration_ms / 1000.0))
        total_samples = samples_per_channel * num_channels

        audio_data_np = np.zeros(total_samples, dtype=np.int16)

        if total_samples > 10:
            audio_data_np[5] = 1000
            audio_data_np[10] = -500

        return TranscriptionRequest(
            data=audio_data_np.tobytes(),
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
            language=language,
        )

    def test_predict_mono_16khz(self):
        audio_request = self._create_dummy_audio_request(
            sample_rate=16000, num_channels=1
        )
        transcription = self.api.predict(audio_request)
        self.assertEqual(transcription, "test transcription")
        self.mock_model.transcribe.assert_called_once()
        called_audio_arg = self.mock_model.transcribe.call_args[0][0]
        self.assertIsInstance(called_audio_arg, np.ndarray)
        self.assertEqual(called_audio_arg.ndim, 1)

    def test_predict_stereo_16khz(self):
        audio_request = self._create_dummy_audio_request(
            sample_rate=16000, num_channels=2
        )
        transcription = self.api.predict(audio_request)
        self.assertEqual(transcription, "test transcription")
        called_audio_arg = self.mock_model.transcribe.call_args[0][0]
        self.assertEqual(called_audio_arg.ndim, 1)

    def test_predict_stereo_48khz_resample(self):
        audio_request = self._create_dummy_audio_request(
            sample_rate=48000, num_channels=2
        )
        transcription = self.api.predict(audio_request)
        self.assertEqual(transcription, "test transcription")
        called_audio_arg = self.mock_model.transcribe.call_args[0][0]
        self.assertEqual(called_audio_arg.ndim, 1)

        expected_resampled_len = int(16000 * (100 / 1000.0))
        self.assertEqual(len(called_audio_arg), expected_resampled_len)

    def test_predict_stereo_odd_length_padding(self):

        sample_rate = 16000
        num_channels = 2
        duration_ms = 100
        samples_per_channel = int(sample_rate * (duration_ms / 1000.0))

        audio_data_np = np.zeros(samples_per_channel * num_channels - 1, dtype=np.int16)
        audio_data_np[5] = 1000

        audio_request = TranscriptionRequest(
            data=audio_data_np.tobytes(),
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel - 1,
            language="vi",
        )
        with patch("local_apis.fasterwhisperapi.logger") as mock_logger:
            transcription = self.api.predict(audio_request)
            self.assertEqual(transcription, "test transcription")
            mock_logger.warning.assert_called_once()

    def test_predict_silent_audio_normalization(self):

        audio_data_np = np.zeros(16000 * 1, dtype=np.int16)
        audio_request = TranscriptionRequest(
            data=audio_data_np.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=16000,
        )
        transcription = self.api.predict(audio_request)
        self.assertEqual(transcription, "test transcription")
        called_audio_arg = self.mock_model.transcribe.call_args[0][0]
        self.assertTrue(np.all(called_audio_arg == 0))

    def test_encode_response(self):
        response = self.api.encode_response("hello world")
        self.assertIsInstance(response, TranscriptionResponse)
        self.assertEqual(response.transcription, "hello world")


if __name__ == "__main__":
    unittest.main()
