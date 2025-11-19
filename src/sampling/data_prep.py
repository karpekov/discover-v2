#!/usr/bin/env python3
"""
Automatic data preparation utility for training pipeline.

This utility automatically generates all required data for training:
1. Data sampling (FL/FD with and without presegmentation)
2. Caption generation (baseline/sourish)
3. Text embedding generation (any encoder)

Programmatic Usage:
    from sampling.data_prep import DataPreparer

    preparer = DataPreparer()
    preparer.prepare_all_data(config)

Command Line Usage:
    # Prepare data for a training config
    python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml

    # Prepare only non-presegmented version
    python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml --no-presegmented

    # Verbose logging
    python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml --verbose
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class DataPreparer:
    """Automatic data preparation for training pipeline."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize data preparer.

        Args:
            project_root: Root directory of project. If None, infers from file location.
        """
        if project_root is None:
            # Assume we're in src/sampling, go up two levels to project root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.logger = logging.getLogger(__name__)

        # Script paths
        self.sample_script = self.project_root / "sample_data.py"
        self.caption_script = self.project_root / "src" / "captions" / "generate_captions.py"
        self.embedding_script = self.project_root / "src" / "text_encoders" / "encode_captions.py"
        self.configs_dir = self.project_root / "configs"

    def parse_data_requirements(self, config) -> Dict[str, any]:
        """Parse config to determine data requirements.

        Returns:
            Dict with:
                - dataset_name: str (milan, aruba, etc.)
                - sampling_strategy: str (FL_20, FD_60, etc.)
                - is_presegmented: bool
                - caption_style: str (baseline, sourish)
                - text_encoder: str (clip, gte, minilm, etc.)
                - splits: List[str] (train, val, test)
        """
        requirements = {
            'dataset_name': None,
            'sampling_strategy': None,
            'is_presegmented': False,
            'caption_style': 'baseline',
            'text_encoder': None,
            'splits': ['train', 'val', 'test']
        }

        # Parse from train_data_path
        # Expected format: data/processed/casas/{dataset}/{strategy}/train.json
        train_path = Path(config.train_data_path)
        parts = train_path.parts

        # Find dataset name (after 'casas' or 'marble')
        for i, part in enumerate(parts):
            if part in ['casas', 'marble'] and i + 1 < len(parts):
                requirements['dataset_name'] = parts[i + 1]
                if i + 2 < len(parts):
                    requirements['sampling_strategy'] = parts[i + 2]
                break

        # Check if presegmented
        if requirements['sampling_strategy'] and '_p' in requirements['sampling_strategy']:
            requirements['is_presegmented'] = True

        # Parse caption style from embeddings path
        # Expected format: train_embeddings_{caption_style}_{encoder}.npz
        if hasattr(config, 'train_text_embeddings_path') and config.train_text_embeddings_path:
            emb_path = Path(config.train_text_embeddings_path)
            filename = emb_path.stem  # Remove .npz

            # Parse: train_embeddings_baseline_clip -> baseline, clip
            parts = filename.split('_')
            if 'embeddings' in parts:
                emb_idx = parts.index('embeddings')
                # Everything between 'embeddings' and last part is caption style
                if emb_idx + 2 < len(parts):
                    requirements['caption_style'] = '_'.join(parts[emb_idx + 1:-1])
                    requirements['text_encoder'] = parts[-1]
                elif emb_idx + 1 < len(parts):
                    # Just encoder, default caption style
                    requirements['text_encoder'] = parts[-1]

        return requirements

    def check_data_exists(self, path: str) -> bool:
        """Check if data file exists."""
        return Path(path).exists() if path else False

    def get_sampling_config_path(self, dataset: str, strategy: str) -> Optional[Path]:
        """Get path to sampling config.

        Args:
            dataset: Dataset name (milan, aruba, etc.)
            strategy: Sampling strategy (FL_20, FD_60, etc.)

        Returns:
            Path to config file or None if not found
        """
        config_path = self.configs_dir / "sampling" / f"{dataset}_{strategy}.yaml"
        if config_path.exists():
            return config_path

        self.logger.warning(f"Sampling config not found: {config_path}")
        return None

    def get_text_encoder_config_path(self, encoder: str) -> Optional[Path]:
        """Get path to text encoder config.

        Args:
            encoder: Encoder name (clip, gte, minilm, etc.)

        Returns:
            Path to config file or None if not found
        """
        # Map short names to config files
        encoder_map = {
            'clip': 'clip_vit_base.yaml',
            'gte': 'gte_base.yaml',
            'minilm': 'minilm_l6.yaml',
            'distilroberta': 'distilroberta_base.yaml',
            'llama': 'llama_embed_8b.yaml',
            'gemma': 'embeddinggemma_300m.yaml',
            'siglip': 'siglip_base.yaml'
        }

        config_name = encoder_map.get(encoder.lower(), f"{encoder}.yaml")
        config_path = self.configs_dir / "text_encoders" / config_name

        if config_path.exists():
            return config_path

        self.logger.warning(f"Text encoder config not found: {config_path}")
        return None

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and log output.

        Args:
            cmd: Command to run as list of strings
            description: Description for logging

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"{description}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.warning(result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with exit code {e.returncode}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            return False

    def prepare_sampling(self, dataset: str, strategy: str) -> bool:
        """Prepare sampled data.

        Args:
            dataset: Dataset name (milan, aruba, etc.)
            strategy: Sampling strategy (FL_20, FD_60, etc.)

        Returns:
            True if successful or data already exists
        """
        # Check if data already exists
        data_dir = self.project_root / "data" / "processed" / "casas" / dataset / strategy
        train_file = data_dir / "train.json"

        if train_file.exists():
            self.logger.info(f"✓ Sampled data already exists: {data_dir}")
            return True

        # Get sampling config
        config_path = self.get_sampling_config_path(dataset, strategy)
        if not config_path:
            self.logger.error(f"Cannot find sampling config for {dataset}_{strategy}")
            return False

        # Run sampling
        cmd = [
            sys.executable,
            str(self.sample_script),
            "--config", str(config_path)
        ]

        success = self.run_command(
            cmd,
            f"Generating sampled data: {dataset}_{strategy}"
        )

        if success and train_file.exists():
            self.logger.info(f"✓ Successfully generated: {data_dir}")
            return True
        else:
            self.logger.error(f"Failed to generate sampled data: {data_dir}")
            return False

    def prepare_captions(self, dataset: str, strategy: str, caption_style: str,
                        splits: List[str] = None) -> bool:
        """Prepare captions for all splits.

        Args:
            dataset: Dataset name (milan, aruba, etc.)
            strategy: Sampling strategy (FL_20, FD_60, etc.)
            caption_style: Caption style (baseline, sourish, etc.)
            splits: List of splits to generate (default: ['train', 'val', 'test'])

        Returns:
            True if successful or captions already exist
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        data_dir = self.project_root / "data" / "processed" / "casas" / dataset / strategy

        # Check if captions already exist
        caption_files_exist = all(
            (data_dir / f"{split}_captions_{caption_style}.json").exists()
            for split in splits
        )

        if caption_files_exist:
            self.logger.info(f"✓ Captions already exist: {data_dir} ({caption_style})")
            return True

        # Run caption generation
        cmd = [
            sys.executable,
            str(self.caption_script),
            "--data-dir", str(data_dir),
            "--caption-style", caption_style,
            "--dataset-name", dataset,
            "--split", "all"
        ]

        success = self.run_command(
            cmd,
            f"Generating {caption_style} captions: {dataset}_{strategy}"
        )

        if success:
            self.logger.info(f"✓ Successfully generated captions: {data_dir}")
            return True
        else:
            self.logger.error(f"Failed to generate captions: {data_dir}")
            return False

    def prepare_embeddings(self, dataset: str, strategy: str, caption_style: str,
                          text_encoder: str, splits: List[str] = None) -> bool:
        """Prepare text embeddings for all splits.

        Args:
            dataset: Dataset name (milan, aruba, etc.)
            strategy: Sampling strategy (FL_20, FD_60, etc.)
            caption_style: Caption style (baseline, sourish, etc.)
            text_encoder: Text encoder (clip, gte, minilm, etc.)
            splits: List of splits to generate (default: ['train', 'val', 'test'])

        Returns:
            True if successful or embeddings already exist
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        data_dir = self.project_root / "data" / "processed" / "casas" / dataset / strategy

        # Check if embeddings already exist
        embedding_files_exist = all(
            (data_dir / f"{split}_embeddings_{caption_style}_{text_encoder}.npz").exists()
            for split in splits
        )

        if embedding_files_exist:
            self.logger.info(f"✓ Embeddings already exist: {data_dir} ({text_encoder})")
            return True

        # Get encoder config
        encoder_config = self.get_text_encoder_config_path(text_encoder)
        if not encoder_config:
            self.logger.error(f"Cannot find text encoder config for: {text_encoder}")
            return False

        # Run embedding generation
        cmd = [
            sys.executable,
            str(self.embedding_script),
            "--data-dir", str(data_dir),
            "--caption-style", caption_style,
            "--split", "all",
            "--config", str(encoder_config)
        ]

        success = self.run_command(
            cmd,
            f"Generating {text_encoder} embeddings: {dataset}_{strategy}"
        )

        if success:
            self.logger.info(f"✓ Successfully generated embeddings: {data_dir}")
            return True
        else:
            self.logger.error(f"Failed to generate embeddings: {data_dir}")
            return False

    def prepare_all_data(self, config, include_presegmented: bool = True) -> bool:
        """Prepare all required data for training.

        Args:
            config: AlignmentConfig object
            include_presegmented: Also generate presegmented version

        Returns:
            True if all data preparation succeeded
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("AUTOMATIC DATA PREPARATION")
        self.logger.info("="*80)

        # Parse requirements from config
        reqs = self.parse_data_requirements(config)

        self.logger.info(f"\nParsed requirements:")
        self.logger.info(f"  Dataset: {reqs['dataset_name']}")
        self.logger.info(f"  Sampling: {reqs['sampling_strategy']}")
        self.logger.info(f"  Presegmented: {reqs['is_presegmented']}")
        self.logger.info(f"  Caption style: {reqs['caption_style']}")
        self.logger.info(f"  Text encoder: {reqs['text_encoder']}")
        self.logger.info(f"  Splits: {reqs['splits']}")

        # Validate requirements
        if not reqs['dataset_name'] or not reqs['sampling_strategy']:
            self.logger.error("Could not parse dataset/sampling strategy from config")
            return False

        if not reqs['text_encoder']:
            self.logger.error("Could not parse text encoder from config")
            return False

        strategies_to_prepare = [reqs['sampling_strategy']]

        # Add presegmented version if requested and not already presegmented
        if include_presegmented and not reqs['is_presegmented']:
            preseg_strategy = reqs['sampling_strategy'] + '_p'
            strategies_to_prepare.append(preseg_strategy)
            self.logger.info(f"\nWill also prepare presegmented version: {preseg_strategy}")

        # Prepare data for each strategy
        all_success = True
        for strategy in strategies_to_prepare:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"PREPARING: {reqs['dataset_name']}_{strategy}")
            self.logger.info(f"{'='*80}")

            # Step 1: Sampling
            success = self.prepare_sampling(reqs['dataset_name'], strategy)
            if not success:
                self.logger.error(f"Failed to prepare sampling for {strategy}")
                all_success = False
                continue

            # Step 2: Captions
            success = self.prepare_captions(
                reqs['dataset_name'],
                strategy,
                reqs['caption_style'],
                reqs['splits']
            )
            if not success:
                self.logger.error(f"Failed to prepare captions for {strategy}")
                all_success = False
                continue

            # Step 3: Embeddings
            success = self.prepare_embeddings(
                reqs['dataset_name'],
                strategy,
                reqs['caption_style'],
                reqs['text_encoder'],
                reqs['splits']
            )
            if not success:
                self.logger.error(f"Failed to prepare embeddings for {strategy}")
                all_success = False
                continue

            self.logger.info(f"\n✓ All data prepared for {strategy}")

        if all_success:
            self.logger.info("\n" + "="*80)
            self.logger.info("DATA PREPARATION COMPLETE!")
            self.logger.info("="*80)
        else:
            self.logger.warning("\n" + "="*80)
            self.logger.warning("DATA PREPARATION COMPLETED WITH ERRORS")
            self.logger.warning("="*80)

        return all_success


def prepare_data_for_config(config_path: str, include_presegmented: bool = True) -> bool:
    """Convenience function to prepare data from config file path.

    Args:
        config_path: Path to alignment config YAML
        include_presegmented: Also generate presegmented version

    Returns:
        True if successful
    """
    from alignment.config import AlignmentConfig

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = AlignmentConfig.from_yaml(config_path)

    # Prepare data
    preparer = DataPreparer()
    return preparer.prepare_all_data(config, include_presegmented)


if __name__ == '__main__':
    """Command line interface for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Automatic data preparation for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data for training config (includes presegmented version)
  python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml

  # Prepare only non-presegmented version
  python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml --no-presegmented

  # Verbose logging
  python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml --verbose

What it does:
  1. Parses your training config to understand data requirements
  2. Checks if required data exists
  3. Automatically runs:
     - Data sampling (sample_data.py)
     - Caption generation (generate_captions.py)
     - Text embedding generation (encode_captions.py)
  4. Generates both regular and presegmented versions (unless --no-presegmented)
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to alignment config YAML'
    )

    parser.add_argument(
        '--no-presegmented',
        action='store_true',
        help='Skip generating presegmented version'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run data preparation
    success = prepare_data_for_config(
        args.config,
        include_presegmented=not args.no_presegmented
    )

    if success:
        print("\n✓ Data preparation completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Data preparation failed!")
        sys.exit(1)

