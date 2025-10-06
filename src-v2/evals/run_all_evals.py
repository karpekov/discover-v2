#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""
Comprehensive evaluation runner for trained HAR models.
Runs all evaluation scripts in sequence to generate a complete analysis suite.

Example usage:
    # Run full evaluation suite (from src-v2 directory)
    cd /Users/alexkarpekov/code/har/casas_clustering/src-v2 && python evals/run_all_evals.py \
        --checkpoint trained_models/milan_tiny_50/best_model.pt \
        --train_data ../data/data_for_alignment/milan_training_50/milan_train.json \
        --test_data ../data/data_for_alignment/milan_training_50/milan_presegmented_test.json \
        --vocab ../data/data_for_alignment/milan_training_50/milan_vocab.json \
        --output_dir analysis/milan_tiny_50

    # Skip specific evaluations
    python run_all_evals.py [args] --skip_visualization --skip_caption_alignment
"""

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime


class ComprehensiveEvaluator:
    """Run all evaluation scripts for a trained model."""

    def __init__(self, config: dict):
        self.config = config
        self.scripts_dir = Path(__file__).parent
        self.start_time = time.time()

        # Convert paths for individual scripts that run from project root
        self._convert_paths_for_project_root()

    def _convert_paths_for_project_root(self):
        """Convert relative paths to work from project root directory."""
        # Convert checkpoint path - needs to be prefixed with src-v2/ since individual scripts run from project root
        if not Path(self.config['checkpoint']).is_absolute():
            checkpoint_path = Path(self.config['checkpoint'])
            if not str(checkpoint_path).startswith('src-v2/'):
                self.config['checkpoint'] = f"src-v2/{self.config['checkpoint']}"

        # Convert data paths - these are relative to src-v2, but individual scripts run from project root
        # So ../data/... from src-v2 becomes data/... from project root
        for path_key in ['train_data', 'test_data', 'vocab']:
            if path_key in self.config and not Path(self.config[path_key]).is_absolute():
                path_str = self.config[path_key]
                if path_str.startswith('../'):
                    # Remove the ../ prefix since we're now running from project root
                    self.config[path_key] = path_str[3:]

        # Convert output directory - should be prefixed with src-v2/ since scripts run from project root
        if 'output_dir' in self.config and not Path(self.config['output_dir']).is_absolute():
            output_dir = self.config['output_dir']
            if not output_dir.startswith('src-v2/'):
                self.config['output_dir'] = f"src-v2/{output_dir}"

    def run_command(self, cmd: list, description: str) -> bool:
        """Run a command and handle errors."""
        print(f"\n{'='*80}")
        print(f"🚀 {description}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")
        print()

        try:
            # Run the command from the project root
            project_root = self.scripts_dir.parent.parent  # Go up two levels from evals/ to project root
            result = subprocess.run(cmd, check=True, cwd=project_root)

            elapsed = time.time() - self.start_time
            print(f"\n✅ {description} completed successfully! ({elapsed/60:.1f}m total elapsed)")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} failed with exit code {e.returncode}")
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"\n❌ {description} failed with error: {e}")
            return False

    def run_embedding_evaluation(self) -> bool:
        """Run dual embedding evaluation (filtered vs unfiltered)."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/evaluate_embeddings.py',
            '--checkpoint', self.config['checkpoint'],
            '--train_data', self.config['train_data'],
            '--test_data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'embeddings_evaluation'),
            '--max_samples', str(self.config['max_samples']),
            '--compare_filtering'  # Always use dual evaluation
        ]

        return self.run_command(cmd, "Embedding Evaluation (Dual: Filtered vs Unfiltered)")

    def run_embedding_visualization(self) -> bool:
        """Run embedding visualization with t-SNE."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/visualize_embeddings.py',
            '--checkpoint', self.config['checkpoint'],
            '--train_data', self.config['train_data'],
            '--test_data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'embedding_visualization'),
            '--max_samples', str(self.config['max_samples']),
            '--split', 'test',
            '--method', 'tsne',
            '--use_matplotlib',
            '--filter_noisy_labels'
        ]

        return self.run_command(cmd, "Embedding Visualization (t-SNE)")

    def run_clustering_visualization(self) -> bool:
        """Run embedding visualization with clustering analysis."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/visualize_embeddings.py',
            '--checkpoint', self.config['checkpoint'],
            '--train_data', self.config['train_data'],
            '--test_data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'clustering_visualization'),
            '--max_samples', str(self.config['max_samples']),
            '--split', 'test',
            '--method', 'tsne',
            '--use_matplotlib',
            '--filter_noisy_labels',
            '--include_clustering',
            '--n_clusters', str(self.config.get('n_clusters', 25))
        ]

        return self.run_command(cmd, "Clustering Visualization (t-SNE with K-means)")

    def run_embedding_alignment(self) -> bool:
        """Run embedding alignment analysis."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/embedding_alignment_analysis.py',
            '--checkpoint', self.config['checkpoint'],
            '--data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'embedding_alignment'),
            '--max_samples', str(self.config['max_samples'])
        ]

        return self.run_command(cmd, "Embedding Alignment Analysis")

    def run_caption_alignment(self) -> bool:
        """Run caption alignment analysis."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/caption_alignment_analysis.py',
            '--checkpoint', self.config['checkpoint'],
            '--data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'caption_alignment'),
            '--max_samples', str(self.config['max_samples']),
            '--filter_noisy'
        ]

        return self.run_command(cmd, "Caption Alignment Analysis")

    def run_clustering_evaluation(self) -> bool:
        """Run clustering evaluation with K-means and DBSCAN."""
        cmd = [
            'conda', 'run', '-n', 'har_env', 'python', 'src-v2/evals/clustering_evaluation.py',
            '--checkpoint', self.config['checkpoint'],
            '--test_data', self.config['test_data'],
            '--vocab', self.config['vocab'],
            '--output_dir', str(Path(self.config['output_dir']) / 'clustering_evaluation'),
            '--max_samples', str(self.config['max_samples']),
            '--n_clusters', str(self.config.get('n_clusters', 50)),
            '--filter_noisy_labels'
        ]

        return self.run_command(cmd, "Clustering Evaluation")

    def create_summary_report(self, results: dict) -> None:
        """Create a summary report of all evaluations."""
        output_dir = Path(self.config['output_dir'])
        report_path = output_dir / 'evaluation_summary.txt'

        total_time = time.time() - self.start_time

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {self.config['checkpoint']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max Samples: {self.config['max_samples']}\n")
            f.write(f"Total Time: {total_time/60:.1f} minutes\n\n")

            f.write("EVALUATION RESULTS:\n")
            f.write("-" * 40 + "\n")

            for eval_name, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                f.write(f"{eval_name:<35} {status}\n")

            f.write(f"\nOUTPUT DIRECTORY: {output_dir}\n")

            # List generated files
            f.write("\nGENERATED FILES:\n")
            f.write("-" * 40 + "\n")

            for subdir in ['embeddings_evaluation', 'embedding_visualization', 'clustering_visualization',
                          'embedding_alignment', 'caption_alignment', 'clustering_evaluation']:
                subdir_path = output_dir / subdir
                if subdir_path.exists():
                    f.write(f"\n{subdir}/\n")
                    for file_path in sorted(subdir_path.glob('*')):
                        if file_path.is_file():
                            f.write(f"  - {file_path.name}\n")

        print(f"\n📄 Summary report saved: {report_path}")

    def run_all_evaluations(self) -> dict:
        """Run all evaluation pipelines."""
        print("🚀 Starting Comprehensive Model Evaluation Suite")
        print(f"📁 Output Directory: {self.config['output_dir']}")
        print(f"🎯 Max Samples: {self.config['max_samples']}")
        print(f"🤖 Model: {Path(self.config['checkpoint']).name}")

        # Create output directory
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

        # Run all evaluations
        results = {}

        # 1. Embedding Evaluation (Dual)
        if not self.config['skip_flags']['embedding_eval']:
            results['Embedding Evaluation'] = self.run_embedding_evaluation()
        else:
            print("⏭️  Skipping Embedding Evaluation")
            results['Embedding Evaluation'] = True  # Mark as skipped (success)

        # 2. Embedding Visualization
        if not self.config['skip_flags']['visualization']:
            results['Embedding Visualization'] = self.run_embedding_visualization()
        else:
            print("⏭️  Skipping Embedding Visualization")
            results['Embedding Visualization'] = True  # Mark as skipped (success)

        # 3. Clustering Visualization (t-SNE with clusters)
        if not self.config['skip_flags']['clustering_visualization']:
            results['Clustering Visualization'] = self.run_clustering_visualization()
        else:
            print("⏭️  Skipping Clustering Visualization")
            results['Clustering Visualization'] = True  # Mark as skipped (success)

        # 4. Embedding Alignment Analysis
        if not self.config['skip_flags']['embedding_alignment']:
            results['Embedding Alignment'] = self.run_embedding_alignment()
        else:
            print("⏭️  Skipping Embedding Alignment")
            results['Embedding Alignment'] = True  # Mark as skipped (success)

        # 5. Caption Alignment Analysis
        if not self.config['skip_flags']['caption_alignment']:
            results['Caption Alignment'] = self.run_caption_alignment()
        else:
            print("⏭️  Skipping Caption Alignment")
            results['Caption Alignment'] = True  # Mark as skipped (success)

        # 6. Clustering Evaluation
        if not self.config['skip_flags']['clustering_eval']:
            results['Clustering Evaluation'] = self.run_clustering_evaluation()
        else:
            print("⏭️  Skipping Clustering Evaluation")
            results['Clustering Evaluation'] = True  # Mark as skipped (success)

        # Create summary report
        self.create_summary_report(results)

        # Final summary
        total_time = time.time() - self.start_time
        successful = sum(results.values())
        total = len(results)

        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE EVALUATION COMPLETE!")
        print("="*80)
        print(f"✅ Successful: {successful}/{total}")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"📁 Results: {self.config['output_dir']}")

        if successful == total:
            print("🎉 All evaluations completed successfully!")
        else:
            failed = [name for name, success in results.items() if not success]
            print(f"⚠️  Failed evaluations: {', '.join(failed)}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation suite for trained HAR models')

    # Model and data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary file')

    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all evaluation results')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples for evaluation')

    # Optional flags
    parser.add_argument('--skip_embedding_eval', action='store_true',
                       help='Skip embedding evaluation')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip embedding visualization')
    parser.add_argument('--skip_embedding_alignment', action='store_true',
                       help='Skip embedding alignment analysis')
    parser.add_argument('--skip_caption_alignment', action='store_true',
                       help='Skip caption alignment analysis')
    parser.add_argument('--skip_clustering_visualization', action='store_true',
                       help='Skip clustering visualization (t-SNE with clusters)')
    parser.add_argument('--skip_clustering_eval', action='store_true',
                       help='Skip clustering evaluation')
    parser.add_argument('--n_clusters', type=int, default=50,
                       help='Number of clusters for K-means in clustering evaluation')

    args = parser.parse_args()

    # Validate paths
    for path_arg in ['checkpoint', 'train_data', 'test_data', 'vocab']:
        path = getattr(args, path_arg)
        if not Path(path).exists():
            print(f"❌ Error: {path_arg} file not found: {path}")
            sys.exit(1)

    # Configuration
    config = {
        'checkpoint': args.checkpoint,
        'train_data': args.train_data,
        'test_data': args.test_data,
        'vocab': args.vocab,
        'output_dir': args.output_dir,
        'max_samples': args.max_samples,
        'skip_flags': {
            'embedding_eval': args.skip_embedding_eval,
            'visualization': args.skip_visualization,
            'clustering_visualization': args.skip_clustering_visualization,
            'embedding_alignment': args.skip_embedding_alignment,
            'caption_alignment': args.skip_caption_alignment,
            'clustering_eval': args.skip_clustering_eval,
        },
        'n_clusters': args.n_clusters
    }

    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator(config)
    results = evaluator.run_all_evaluations()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some evaluations failed


if __name__ == "__main__":
    main()
