# Minority Game RL Training System

A C++ implementation of reinforcement learning agents for the minority game, providing a comprehensive training framework with multiple agent types and evaluation tools.

## Overview

This system allows you to train and evaluate reinforcement learning agents in the minority game environment. The minority game is a model from econophysics where agents compete by choosing between two options, with the minority side winning each round.

## Features

- **Multiple RL Algorithms**: Random, Q-Learning, and DQN agents
- **Single and Multi-Agent Training**: Train individual agents or multiple agents simultaneously
- **Parameter Sweep Training**: Train agents across multiple player configurations with automated evaluation and CSV export
- **Comprehensive Evaluation**: Performance metrics, model persistence, and comparison tools
- **Flexible Configuration**: Command-line interface with extensive customization options
- **Integration with Existing Code**: Built on top of the existing C++ minority game implementation

## Building the System

### Prerequisites

- C++17 compatible compiler (GCC 7+ or Clang 5+)
- Make build system
- Standard C++ libraries

### Compilation

\`\`\`bash
# Build the training executable
make all

# Create necessary directories
make setup

# Run basic tests
make test
\`\`\`

## Usage

### Basic Training

Train a single Q-Learning agent:
\`\`\`bash
./train --agent qlearning --episodes 1000 --players 101 --memory 3
\`\`\`

Train a DQN agent:
\`\`\`bash
./train --agent dqn --episodes 1000 --lr 0.001 --epsilon 1.0
\`\`\`

### Multi-Agent Training

Train multiple agents simultaneously:
\`\`\`bash
./train --multiagent 3 --episodes 500 --players 101
\`\`\`

### Agent Comparison

Compare different agent types:
\`\`\`bash
./train --compare --episodes 200
\`\`\`

### Model Evaluation

Evaluate a saved model:
\`\`\`bash
./train --evaluate models/qlearning_agent_20241201_143022.model --agent qlearning
\`\`\`

### Parameter Sweep Training

Train agents across multiple player configurations with automated evaluation and CSV export:

\`\`\`bash
# Basic sweep across different player counts
./train --sweep "101,301,501" --memory 3 --agent qlearning

# Advanced sweep with custom parameters
./train --sweep "101,201,301,401,501" --memory 3 --agent qlearning \
        --sweep-episodes 2000 --eval-episodes 1000 \
        --lr 0.05 --epsilon 0.15 --output-csv my_results.csv

# Sweep with DQN agent
./train --sweep "101,301,501" --memory 3 --agent dqn --lr 0.001
\`\`\`

The sweep mode:
1. **Trains** the agent for each player configuration
2. **Evaluates** performance over a fixed number of episodes (separate from training)
3. **Exports** comprehensive results to CSV with evaluation metrics
4. **Saves** models with descriptive filenames (e.g., `qlearning_agent_m3_n101.model`)

### Command Line Options

- `--agent TYPE`: Agent type (random, qlearning, dqn) [default: qlearning]
- `--episodes N`: Number of training episodes [default: 1000]
- `--players N`: Number of players in the game [default: 101]
- `--memory N`: Memory size [default: 3]
- `--lr RATE`: Learning rate [default: 0.1]
- `--epsilon EPS`: Exploration rate [default: 0.1]
- `--gamma GAMMA`: Discount factor [default: 0.95]
- `--seed N`: Random seed [default: random]
- `--multiagent N`: Train N RL agents simultaneously
- `--compare`: Compare different agent types
- `--evaluate FILE`: Evaluate a saved model
- `--sweep PLAYERS`: Comma-separated list of player counts for parameter sweep
- `--sweep-episodes N`: Training episodes per sweep configuration [default: 1000]
- `--eval-episodes N`: Evaluation episodes per configuration [default: 500]
- `--output-csv FILE`: CSV filename for sweep results [default: auto-generated]
- `--verbose`: Enable verbose output [default: true]
- `--help`: Show help message

## Architecture

### Core Components

1. **Environment Interface** (`minority_game_env.h/cpp`):
   - `MinorityGameEnv`: Single agent environment
   - `MultiAgentMinorityGameEnv`: Multi-agent environment
   - Observation and action space management

2. **RL Agents** (`rl_agents.h/cpp`):
   - `RandomAgent`: Baseline random agent
   - `QLearningAgent`: Tabular Q-Learning implementation
   - `DQNAgent`: Deep Q-Network (simplified tabular version)

3. **Training Framework** (`training_framework.h/cpp`):
   - `SingleAgentTrainer`: Single agent training and evaluation
   - `MultiAgentTrainer`: Multi-agent training coordination
   - `AgentComparator`: Performance comparison utilities

4. **Integration Layer**:
   - Seamless integration with existing minority game code
   - Preserved original game mechanics and parameters

### Key Features

- **Observation Space**: Binary history of last M game outcomes
- **Action Space**: Binary choice (0 or 1) representing game sides
- **Reward Function**: +1 for minority choice, -1 for majority, with attendance penalty
- **Model Persistence**: Save and load trained models
- **Metrics Tracking**: Comprehensive performance monitoring

## Examples

### Quick Start Examples

\`\`\`bash
# Run all examples
make examples

# Train a Q-Learning agent for 200 episodes
./train --agent qlearning --episodes 200 --players 51 --memory 2

# Train a DQN agent with custom parameters
./train --agent dqn --episodes 200 --lr 0.001 --epsilon 1.0 --memory 2

# Multi-agent training with 2 agents
./train --multiagent 2 --episodes 100 --players 51

# Compare all agent types
./train --compare --episodes 100
\`\`\`

### Advanced Usage

\`\`\`bash
# Train with specific seed for reproducibility
./train --agent qlearning --episodes 500 --seed 42

# Train with custom hyperparameters
./train --agent qlearning --lr 0.05 --epsilon 0.2 --gamma 0.99

# Evaluate multiple models
./train --evaluate models/agent1.model --agent qlearning
./train --evaluate models/agent2.model --agent dqn

# Parameter sweep with reproducible results
./train --sweep "51,101,201,301" --memory 3 --seed 42 --agent qlearning

# Large-scale sweep with extended training and evaluation
./train --sweep "101,201,301,401,501,601,701,801,901,1001" \
        --memory 4 --sweep-episodes 5000 --eval-episodes 2000 \
        --agent qlearning --lr 0.03 --epsilon 0.1
\`\`\`

## Output Files

The system generates several types of output files:

- **Models**: `models/[agent_type]_agent_[timestamp].model`
- **Metrics**: `metrics/[agent_type]_metrics_[timestamp].csv`
- **Comparisons**: `metrics/agent_comparison_[timestamp].csv`
- **Sweep Results**: `sweep_results_[timestamp].csv` (or custom filename)
- **Sweep Models**: `models/[agent_type]_agent_m[memory]_n[players].model`

## Performance Metrics

The system tracks various performance metrics:

- Episode rewards and win rates
- Running averages and standard deviations
- Training time and convergence statistics
- Agent-specific metrics (Q-table size, exploration rate, etc.)
- **Sweep Metrics**: For each configuration, exports evaluation-based metrics including:
  - Average reward and standard deviation (from evaluation episodes)
  - Win rate and standard deviation (from evaluation episodes)
  - Memory size, player count, and hyperparameters
  - Model filename for reproducibility

## Integration Notes

This RL training system is built on top of the existing C++ minority game implementation, preserving:

- Original game mechanics and scoring
- Agent strategy systems
- Random number generation
- Configuration management
- Error handling

The integration maintains backward compatibility while adding comprehensive RL capabilities.

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Ensure C++17 support and all source files are present
2. **Runtime Errors**: Check file permissions for model/metrics directories
3. **Memory Issues**: Reduce episode count or memory capacity for large experiments
4. **Performance**: Use release build (`make release`) for production training

### Debug Mode

Build with debug information:
\`\`\`bash
make debug
./train --agent qlearning --episodes 10 --verbose
\`\`\`

## Contributing

When extending the system:

1. Follow the existing code structure and naming conventions
2. Add appropriate error handling and validation
3. Update documentation and examples
4. Test with multiple agent types and configurations

## License

This project follows the same license as the original minority game implementation (GPL v2+).
