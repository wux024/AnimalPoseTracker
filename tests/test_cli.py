import unittest
from unittest.mock import patch

from animalposetracker.cli import main


class CommandDispatcherTests(unittest.TestCase):
    def test_train_forwards_arguments(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["train", "--config", "project/configs/other.yaml"]), 0)
        run.assert_called_once_with(["--config", "project/configs/other.yaml"])

    def test_val_selects_validation_only_mode(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["val", "--config", "project/configs/other.yaml", "--weights", "best.pt"]), 0)
        run.assert_called_once_with([
            "--validate-only", "--config", "project/configs/other.yaml", "--weights", "best.pt",
        ])

    def test_val_help_is_specific_to_validation(self):
        with patch("sys.stdout") as output:
            with self.assertRaises(SystemExit) as raised:
                main(["val", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_predict_and_export_dispatch(self):
        with patch("animalposetracker.prediction.cli.run", return_value=7) as predict:
            self.assertEqual(main(["predict", "--weights", "best.pt"]), 7)
        predict.assert_called_once_with(["--weights", "best.pt"])

        with patch("animalposetracker.export.cli.run", return_value=0) as export:
            self.assertEqual(main(["export", "--weights", "best.pt", "--format", "onnx"]), 0)
        export.assert_called_once_with(["--weights", "best.pt", "--format", "onnx"])

    def test_no_arguments_prints_help(self):
        with patch("sys.stdout"):
            self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
