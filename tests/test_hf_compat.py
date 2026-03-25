from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keelnet.hf_compat import trainer_processing_kwargs


class TrainerCompatTest(unittest.TestCase):
    def test_prefers_processing_class_when_available(self):
        class NewTrainer:
            def __init__(self, *, model=None, processing_class=None):
                self.model = model
                self.processing_class = processing_class

        kwargs = trainer_processing_kwargs(NewTrainer, "processor")

        self.assertEqual(kwargs, {"processing_class": "processor"})

    def test_falls_back_to_tokenizer_for_older_trainers(self):
        class OldTrainer:
            def __init__(self, *, model=None, tokenizer=None):
                self.model = model
                self.tokenizer = tokenizer

        kwargs = trainer_processing_kwargs(OldTrainer, "processor")

        self.assertEqual(kwargs, {"tokenizer": "processor"})

    def test_returns_empty_mapping_when_neither_parameter_exists(self):
        class MinimalTrainer:
            def __init__(self, *, model=None):
                self.model = model

        kwargs = trainer_processing_kwargs(MinimalTrainer, "processor")

        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
