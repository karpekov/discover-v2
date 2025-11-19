#!/usr/bin/env python3
"""
Unified training script for the discover-v2 pipeline.

This script orchestrates the entire training pipeline:
1. Sample data (optional, if not already sampled)
2. Generate captions (optional, if not already generated)
3. Encode text embeddings (optional, if not already encoded)
4. Train alignment model

Usage:
  # Train with existing data
  python train.py --config configs/alignment/milan_baseline.yaml

  # Train with full pipeline from scratch
  python train.py --config configs/alignment/milan_baseline.yaml --run-full-pipeline

  # Train alignment only (default)
  python train.py --config configs/alignment/milan_baseline.yaml

  # Resume from checkpoint
  python train.py --config configs/alignment/milan_baseline.yaml --resume trained_models/milan/alignment_baseline/checkpoint_step_5000.pt
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.alignment.config import AlignmentConfig
from src.alignment.trainer import AlignmentTrainer
from src.sampling.data_prep import DataPreparer


def setup_logging():
    """Setup basic logging for the main script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def check_data_exists(path: str) -> bool:
    """Check if data file exists."""
    return Path(path).exists() if path else False


def run_data_sampling(config: AlignmentConfig, logger: logging.Logger):
    """Run data sampling step (Step 1)."""
    logger.info("=" * 80)
    logger.info("STEP 1: Data Sampling")
    logger.info("=" * 80)

    # This requires a sampling config - for now, we'll skip auto-sampling
    # Users should run sample_data.py separately
    logger.warning("Automatic data sampling not yet implemented in unified pipeline.")
    logger.warning("Please run `python sample_data.py --config <sampling_config>` separately.")

    return False


def run_caption_generation(config: AlignmentConfig, logger: logging.Logger):
    """Run caption generation step (Step 3)."""
    logger.info("=" * 80)
    logger.info("STEP 3: Caption Generation")
    logger.info("=" * 80)

    # Check if captions exist
    if config.train_captions_path and check_data_exists(config.train_captions_path):
        logger.info(f"Captions already exist at: {config.train_captions_path}")
        return True

    # This requires a caption config - for now, we'll skip auto-generation
    logger.warning("Automatic caption generation not yet implemented in unified pipeline.")
    logger.warning("Please run `python generate_captions.py --config <caption_config>` separately.")

    return False


def run_text_encoding(config: AlignmentConfig, logger: logging.Logger):
    """Run text encoding step (Step 4)."""
    logger.info("=" * 80)
    logger.info("STEP 4: Text Encoding")
    logger.info("=" * 80)

    # Check if text embeddings exist
    if config.train_text_embeddings_path and check_data_exists(config.train_text_embeddings_path):
        logger.info(f"Text embeddings already exist at: {config.train_text_embeddings_path}")
        return True

    # If captions exist and text encoder config is provided, encode them
    if config.train_captions_path and config.text_encoder_config_path:
        if check_data_exists(config.train_captions_path):
            logger.info(f"Encoding captions from: {config.train_captions_path}")

            try:
                from encode_captions import encode_captions

                # Encode training captions
                logger.info("Encoding training captions...")
                train_output_path = encode_captions(
                    caption_file=config.train_captions_path,
                    encoder_config=config.text_encoder_config_path,
                    output_path=None,  # Auto-generate
                    device='auto',
                    batch_size=64
                )
                logger.info(f"Saved training embeddings to: {train_output_path}")

                # Update config with generated embeddings path
                config.train_text_embeddings_path = train_output_path

                # Encode validation captions if provided
                if config.val_captions_path and check_data_exists(config.val_captions_path):
                    logger.info("Encoding validation captions...")
                    val_output_path = encode_captions(
                        caption_file=config.val_captions_path,
                        encoder_config=config.text_encoder_config_path,
                        output_path=None,  # Auto-generate
                        device='auto',
                        batch_size=64
                    )
                    logger.info(f"Saved validation embeddings to: {val_output_path}")
                    config.val_text_embeddings_path = val_output_path

                return True

            except Exception as e:
                logger.error(f"Error encoding captions: {e}")
                return False
        else:
            logger.warning(f"Captions not found at: {config.train_captions_path}")
            return False

    logger.warning("No text embeddings or captions available for encoding.")
    return False


def run_alignment_training(config: AlignmentConfig, logger: logging.Logger, resume_path: str = None):
    """Run alignment training step (Step 5)."""
    logger.info("=" * 80)
    logger.info("STEP 5: Alignment Training")
    logger.info("=" * 80)

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

    # Create trainer
    logger.info("Initializing alignment trainer...")
    trainer = AlignmentTrainer(config)

    # Load checkpoint if resuming
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    # Train
    logger.info("Starting alignment training...")
    trainer.train()

    logger.info("Alignment training completed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified training script for discover-v2 pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train alignment model with existing data
  python train.py --config configs/alignment/milan_baseline.yaml

  # Train with full pipeline (sample data, generate captions, encode text, train alignment)
  python train.py --config configs/alignment/milan_baseline.yaml --run-full-pipeline

  # Train alignment only (default)
  python train.py --config configs/alignment/milan_baseline.yaml --skip-data-checks

  # Resume from checkpoint
  python train.py --config configs/alignment/milan_baseline.yaml --resume trained_models/milan/alignment_baseline/checkpoint_step_5000.pt
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to alignment config YAML file'
    )

    parser.add_argument(
        '--run-full-pipeline',
        action='store_true',
        help='Run full pipeline: sampling → captions → text encoding → alignment training'
    )

    parser.add_argument(
        '--skip-data-checks',
        action='store_true',
        help='Skip data existence checks and proceed directly to training'
    )

    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Automatically prepare any missing data (sampling, captions, embeddings)'
    )

    parser.add_argument(
        '--no-presegmented',
        action='store_true',
        help='Skip generating presegmented version when preparing data'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("DISCOVER-V2 UNIFIED TRAINING PIPELINE")
    logger.info("=" * 80)

    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = AlignmentConfig.from_yaml(args.config)

    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        logger.info(f"Output directory overridden to: {config.output_dir}")

    # Automatic data preparation if requested
    if args.prepare_data:
        logger.info("Automatic data preparation enabled")
        preparer = DataPreparer()
        success = preparer.prepare_all_data(
            config,
            include_presegmented=not args.no_presegmented
        )
        if not success:
            logger.error("Data preparation failed")
            sys.exit(1)

    # Check if running full pipeline
    if args.run_full_pipeline:
        logger.info("Running full pipeline mode")

        # Step 1: Data sampling
        if not check_data_exists(config.train_data_path):
            logger.info("Training data not found, running data sampling...")
            success = run_data_sampling(config, logger)
            if not success:
                logger.error("Data sampling failed or not implemented. Please run sampling manually.")
                sys.exit(1)
        else:
            logger.info(f"Training data already exists at: {config.train_data_path}")

        # Step 3: Caption generation (if using captions instead of pre-computed embeddings)
        if config.train_captions_path and not config.train_text_embeddings_path:
            if not check_data_exists(config.train_captions_path):
                logger.info("Captions not found, running caption generation...")
                success = run_caption_generation(config, logger)
                if not success:
                    logger.error("Caption generation failed or not implemented. Please run caption generation manually.")
                    sys.exit(1)
            else:
                logger.info(f"Captions already exist at: {config.train_captions_path}")

        # Step 4: Text encoding
        if not config.train_text_embeddings_path or not check_data_exists(config.train_text_embeddings_path):
            logger.info("Text embeddings not found, running text encoding...")
            success = run_text_encoding(config, logger)
            if not success:
                logger.error("Text encoding failed. Please check captions and text encoder config.")
                sys.exit(1)
        else:
            logger.info(f"Text embeddings already exist at: {config.train_text_embeddings_path}")

    # Data checks (unless skipped)
    if not args.skip_data_checks:
        logger.info("Checking data availability...")

        # Check sensor data
        if not check_data_exists(config.train_data_path):
            logger.error(f"Training data not found at: {config.train_data_path}")
            logger.error("Please run data sampling first with: python sample_data.py --config <sampling_config>")
            sys.exit(1)

        # Check text embeddings or captions
        has_embeddings = check_data_exists(config.train_text_embeddings_path) if config.train_text_embeddings_path else False
        has_captions = check_data_exists(config.train_captions_path) if config.train_captions_path else False

        if not (has_embeddings or has_captions):
            logger.error("Neither text embeddings nor captions found.")
            logger.error("Please either:")
            logger.error("  1. Run caption generation: python generate_captions.py --config <caption_config>")
            logger.error("  2. Run text encoding: python encode_captions.py --captions <captions.json> --encoder <encoder_config>")
            sys.exit(1)

        # If only captions exist, offer to encode them
        if has_captions and not has_embeddings and config.text_encoder_config_path:
            logger.info("Captions found but embeddings not found. Encoding captions...")
            success = run_text_encoding(config, logger)
            if not success:
                logger.error("Text encoding failed.")
                sys.exit(1)

        logger.info("All required data is available!")

    # Step 5: Alignment training
    success = run_alignment_training(config, logger, resume_path=args.resume)

    if success:
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {config.output_dir}")
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()

