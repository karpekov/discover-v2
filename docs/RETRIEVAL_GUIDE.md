# Smart-Home Retrieval System Guide

## Score Interpretation

### What the Scores Mean
The retrieval scores are **cosine similarity scores** from FAISS IndexFlatIP (Inner Product):

- **Range**: -1.0 to +1.0 (embeddings are L2-normalized)
- **Higher = Better**: Scores closer to 1.0 indicate stronger semantic similarity
- **Interpretation**:
  - **0.20-0.30**: Strong semantic match (excellent)
  - **0.15-0.20**: Good semantic match (very good)
  - **0.10-0.15**: Moderate match (decent)
  - **0.05-0.10**: Weak match (poor)
  - **< 0.05**: Very weak/unrelated (very poor)

### Example Score Ranges from Training
- **"cooking"**: 0.198-0.199 (strong kitchen activity match)
- **"night wandering"**: 0.147-0.153 (good multi-room night movement)
- **"bathroom activity"**: 0.15-0.25 (typical range for bathroom sequences)

## Enhanced Metadata Available

### 🏷️ Labels & Activity Classification
- **Inferred Activity**: Automatically classified from caption text
  - `cooking`, `bathroom`, `bedroom`, `night_wandering`, `morning_routine`, `general_activity`
- **Caption Day**: Day of week extracted from caption (Monday, Tuesday, etc.)
- **Caption Time**: Time period from caption (morning, evening, night, afternoon)

### 📊 Sequence Metadata
- **Events**: Valid events vs total sequence length (e.g., 20/20)
- **Duration**: Total time span of the sequence in seconds
- **Rooms**: List of unique rooms visited + total count
- **Room Path**: Sequential room transitions (first 5 rooms)
- **Time Period**: Primary time-of-day bucket from sensors
- **Day of Week**: From sensor data (if available)
- **Time Delta**: Bucket for time between events

### 🔧 Detailed Event Information
Each event includes:
- **Sensor ID**: Specific sensor that triggered (e.g., M022, M012)
- **Room ID**: Room location (kitchen, living_room, etc.)
- **Event Type**: ON/OFF sensor activation
- **Sensor Type**: Type of sensor (motion, door, etc.)
- **Time Bucket**: Time-of-day categorization
- **Coordinates**: Normalized (x,y) position in house
- **Time Delta**: Seconds since previous event
- **Event Index**: Position in sequence

## Usage Examples

### Basic Query
```bash
python query_retrieval.py --query "cooking" --top_k 3
```

### Interactive Mode
```bash
python query_retrieval.py --interactive
```

### Demo Mode (Multiple Queries)
```bash
python query_retrieval.py  # Runs demo with common queries
```

## Common Query Types

### Activity-Based Queries
- `"cooking"` - Kitchen activities, meal preparation
- `"bathroom activity"` - Bathroom visits, hygiene routines
- `"night wandering"` - Restless nighttime movement
- `"morning routine"` - Early morning activities
- `"bedroom activity"` - Bedroom-related movements

### Location-Based Queries
- `"kitchen activity"` - Any kitchen-related events
- `"living room movement"` - Living room activities
- `"bedroom visits"` - Bedroom entries/activities

### Time-Based Queries
- `"evening activities"` - Evening time periods
- `"night movement"` - Nighttime activities
- `"morning activities"` - Morning routines

### Pattern-Based Queries
- `"restless movement"` - Multiple room transitions
- `"single room activity"` - Contained activities
- `"quick movements"` - Short duration sequences

## Understanding Results

### High-Quality Matches (Score > 0.15)
- Strong semantic alignment between query and sensor patterns
- Consistent room usage patterns
- Appropriate time-of-day matching
- Coherent activity sequences

### Moderate Matches (Score 0.10-0.15)
- Some semantic similarity
- May have partial pattern matches
- Could be related but not identical activities

### Low Matches (Score < 0.10)
- Weak semantic relationship
- Different activity patterns
- May be noise or unrelated sequences

## Model Performance Insights

### What the Model Learned Well
1. **Room-Activity Associations**: Kitchen→cooking, bathroom→hygiene
2. **Time Patterns**: Night→wandering, evening→cooking
3. **Movement Patterns**: Multi-room vs single-room activities
4. **Sensor Sequences**: Typical ON/OFF patterns for activities

### Limitations
1. **Score Range**: Relatively narrow (0.10-0.25) due to training data size
2. **Activity Diversity**: Limited by caption vocabulary
3. **Temporal Resolution**: Coarse time buckets may miss fine patterns
4. **Sensor Specificity**: Some sensors may be underrepresented

## File Structure
- `query_retrieval.py`: Main retrieval script with enhanced metadata
- `extract_night_wandering.py`: Specialized script for specific queries
- `debug_training.py`: Training debugging utilities
- `eval_retrieval.py`: Comprehensive evaluation metrics
