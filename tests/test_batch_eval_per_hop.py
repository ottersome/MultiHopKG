from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from batch_eval_per_hop import (
    CACHE_SCHEMA_VERSION,
    ReportIdentity,
    build_command,
    reusable_checkpoint,
    run_config,
    training_fingerprint,
)
from src.utils.experiment_io import write_run_manifest


class PerHopRunReuseTests(unittest.TestCase):

    def test_reusable_checkpoint_requires_matching_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            checkpoint = directory / 'model_best.tar'
            checkpoint.touch()
            manifest = directory / 'manifest.json'
            manifest.write_text(json.dumps({
                'schema_version': CACHE_SCHEMA_VERSION,
                'fingerprint': 'expected',
                'completed': True,
                'checkpoint_path': str(checkpoint),
            }))

            self.assertEqual(reusable_checkpoint(manifest, 'expected'), checkpoint)
            self.assertIsNone(reusable_checkpoint(manifest, 'different'))
            checkpoint.unlink()
            self.assertIsNone(reusable_checkpoint(manifest, 'expected'))


    def test_rerun_switches_from_training_to_checkpoint_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            launcher = directory / 'launcher.sh'
            launcher.write_text('#!/usr/bin/env bash\n')
            config_path = directory / 'config.sh'
            config_path.touch()
            checkpoint = directory / 'model_best.tar'
            checkpoint.touch()
            log_dir = directory / 'logs'
            log_dir.mkdir()
            args = argparse.Namespace(
                gpu=0,
                extra_args=[],
                force_retrain=False,
                dry_run=False,
                split='test',
            )
            identity = ReportIdentity('Kinship', 'Single', 'MINERVA', 'Models', 2)

            def complete_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                metrics_path = Path(command[command.index('--metrics_output_path') + 1])
                manifest_path = Path(command[command.index('--run_metadata_output_path') + 1])
                fingerprint = command[command.index('--run_fingerprint') + 1]
                metrics_path.write_text(json.dumps({
                    'test': {'per_hop/2hop_hits@1': 0.75},
                }))
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps({
                    'schema_version': CACHE_SCHEMA_VERSION,
                    'fingerprint': fingerprint,
                    'completed': True,
                    'checkpoint_path': str(checkpoint),
                }))
                return subprocess.CompletedProcess(command, 0, '', '')

            with patch('batch_eval_per_hop.subprocess.run', side_effect=complete_run):
                first = run_config(
                    config_path,
                    identity,
                    12,
                    args,
                    directory,
                    launcher,
                    log_dir,
                    log_dir,
                    {'seed': '12'},
                )
                second = run_config(
                    config_path,
                    identity,
                    12,
                    args,
                    directory,
                    launcher,
                    log_dir,
                    log_dir,
                    {'seed': '12'},
                )
                args.force_retrain = True
                forced = run_config(
                    config_path,
                    identity,
                    12,
                    args,
                    directory,
                    launcher,
                    log_dir,
                    log_dir,
                    {'seed': '12'},
                )

            self.assertFalse(first.reused_checkpoint)
            self.assertIn('--train', first.command)
            self.assertTrue(second.reused_checkpoint)
            self.assertIn('--inference', second.command)
            self.assertEqual(second.hits_at_1, 0.75)
            self.assertFalse(forced.reused_checkpoint)
            self.assertIn('--train', forced.command)


if __name__ == '__main__':
    unittest.main()
