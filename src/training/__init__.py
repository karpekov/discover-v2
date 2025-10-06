"""Training modules for different model types."""

# Don't import everything by default to avoid metadata loading issues
# Import specific trainers only when needed

__all__ = [
    'train_clip',
    'classification_trainer'
]
