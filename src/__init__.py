"""
Dual-Encoder Alignment Pipeline (src)

Complete pipeline for CASAS data processing and dual-encoder alignment.
Implements Recipe R2 for HAR clustering research.

Architecture:
- data/: Complete data processing pipeline
- models/: Model architectures (placeholder)
- alignment/: Dual-encoder training (placeholder)
- experiments/: Experiment configurations and runners
- config/: Configuration management
- utils/: General utilities

Usage:
    from data.pipeline import DualEncoderPipeline
    from data.data_config import ProcessingConfig

    config = ProcessingConfig(dataset_name='milan')
    pipeline = DualEncoderPipeline(config)
    stats = pipeline.process()
"""

__version__ = "2.0.0"
__author__ = "Alex Karpekov"
