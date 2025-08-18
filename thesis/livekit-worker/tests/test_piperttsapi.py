import unittest
from unittest.mock import MagicMock, patch, call
import base64
import io


import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from local_apis.piperttsapi import (
    PiperTextToWavLitAPI,
    TTSRequest,
    TTSResponse,
    VOICE_MODEL_PATH,
    CONFIG_PATH,
)


class TestPiperTextToWavLitAPI(unittest.TestCase):

    def setUp(self):

        self.mock_piper_voice_instance = MagicMock()
        self.mock_piper_voice_instance.synthesize.return_value = None

        self.patcher_voice_load = patch(
            "local_apis.piperttsapi.PiperVoice.load",
            return_value=self.mock_piper_voice_instance,
        )
        self.mock_piper_voice_load = self.patcher_voice_load.start()

        self.api = PiperTextToWavLitAPI()

        with patch(
            "local_apis.piperttsapi.torch.cuda.is_available", return_value=False
        ):
            self.api.setup(device="cpu")

    def tearDown(self):
        self.patcher_voice_load.stop()

    def test_setup_cpu(self):
        with patch(
            "local_apis.piperttsapi.torch.cuda.is_available", return_value=False
        ) as mock_cuda_available:
            self.api.setup(device="cpu")
            self.mock_piper_voice_load.assert_called_with(
                model_path=VOICE_MODEL_PATH, config_path=CONFIG_PATH, use_cuda=False
            )
            self.assertIsNotNone(self.api.voice)

    def test_setup_cuda(self):
        with patch(
            "local_apis.piperttsapi.torch.cuda.is_available", return_value=True
        ) as mock_cuda_available:
            self.api.setup(device="cuda")
            self.mock_piper_voice_load.assert_called_with(
                model_path=VOICE_MODEL_PATH, config_path=CONFIG_PATH, use_cuda=True
            )

    def test_decode_request(self):
        request_data = {"text": "hello world", "length_scale": 0.5}

        tts_request = TTSRequest(**request_data)
        decoded = self.api.decode_request(tts_request)
        self.assertEqual(decoded.text, "hello world")
        self.assertEqual(decoded.length_scale, 0.5)

    def test_predict(self):
        test_text = "This is a test sentence."
        test_length_scale = 0.8
        request_data = TTSRequest(text=test_text, length_scale=test_length_scale)

        dummy_audio_bytes = b"dummy_audio_data"

        def mock_synthesize(text, audio_buffer, length_scale):
            audio_buffer.write(dummy_audio_bytes)

        self.mock_piper_voice_instance.synthesize = MagicMock(
            side_effect=mock_synthesize
        )

        audio_output = self.api.predict(request_data)

        self.mock_piper_voice_instance.synthesize.assert_called_once()

        args, kwargs = self.mock_piper_voice_instance.synthesize.call_args
        self.assertEqual(args[0], test_text)
        self.assertIsInstance(args[1], io.BytesIO)
        self.assertEqual(kwargs["length_scale"], test_length_scale)
        self.assertEqual(audio_output, dummy_audio_bytes)

    def test_encode_response(self):
        dummy_audio_bytes = b"raw_audio_data_for_encoding"
        response = self.api.encode_response(dummy_audio_bytes)
        self.assertIsInstance(response, TTSResponse)
        self.assertEqual(response.content_type, "audio/wav")
        expected_base64 = base64.b64encode(dummy_audio_bytes).decode("utf-8")
        self.assertEqual(response.audio_content, expected_base64)


if __name__ == "__main__":
    unittest.main()
