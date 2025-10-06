
# Converted from legacy .cursorrules format

# Cursor Rules for discover-v2 Project

## Environment and Execution
- Always use `conda activate discover-v2-env` when running Python scripts or commands
- Test imports and functionality with the activated conda environment
- Ensure all dependencies are available in the discover-v2-env environment

## Project Structure
- Primary source code is in the `src/` directory
- Configuration files are in `configs/` (training configs in `configs/training/`, data generation configs in `configs/data_generation/`)
- Data is organized as:
  - `data/raw/` - Raw datasets
  - `data/processed/` - Processed datasets
- Results and outputs go to:
  - `logs/text/` - Text logs
  - `logs/wandb/` - WandB logs
  - `trained_models/` - Model checkpoints
  - `results/evals/` - Evaluation results

## Documentation
- Add documentation to `docs/` folder
- Keep documentation short and to the point
- Documentation is primarily for future Cursor chat sessions, not human consumption
- Update existing docs when making structural changes
- **IMPORTANT**: Always consult existing docs in `docs/` folder before making changes
- **IMPORTANT**: Update relevant docs when making major structural or functional changes

## Testing and Scripts
- Prefer not to create separate testing scripts
- If a test script is necessary, delete it after successful test completion
- Use inline testing commands instead of persistent test files
- Test functionality directly with imports and simple commands

## Code Organization
- Use relative imports within the `src/` directory
- Maintain clean separation between data processing, training, and evaluation
- Follow existing patterns for file organization and naming
- Update all references when making structural changes

## Git and Version Control
- Commit changes with descriptive messages
- Push changes to maintain remote synchronization
- Use meaningful commit messages that describe the purpose of changes
