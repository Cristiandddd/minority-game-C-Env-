/***************************************************************************
                          minority_game_env.h  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#ifndef _MINORITY_GAME_ENV_H_
#define _MINORITY_GAME_ENV_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "minority.h"
#include "agent.h"
#include "configuration.h"

// Forward declarations
class RLAgent;

// Environment state information
struct EnvInfo {
    int step;
    int total_attendance;
    int winning_side;
    double rl_agent_score;
    int rl_agent_wins;
    double win_rate;
    std::vector<int> agent_bets;
    unsigned long memory_state;
    std::vector<int> non_rl_agent_wins;  // Win count for each non-RL agent
    std::vector<double> non_rl_win_rates; // Win rate for each non-RL agent
    
    EnvInfo() : step(0), total_attendance(0), winning_side(0), 
                rl_agent_score(0.0), rl_agent_wins(0), win_rate(0.0), 
                memory_state(0) {}
};

// Observation space for RL agents
struct Observation {
    std::vector<int> history;  // Last memory_size outcomes
    
    Observation(int memory_size) : history(memory_size, 0) {}
    
    void update(const std::vector<int>& game_history, int memory_size) {
        history.clear();
        if ((int)game_history.size() < memory_size) {
            // Pad with zeros if not enough history
            for (int i = 0; i < memory_size - (int)game_history.size(); i++) {
                history.push_back(0);
            }
            for (int outcome : game_history) {
                history.push_back(outcome);
            }
        } else {
            // Take last memory_size outcomes
            for (int i = game_history.size() - memory_size; i < (int)game_history.size(); i++) {
                history.push_back(game_history[i]);
            }
        }
    }
};

// Single agent RL environment
class MinorityGameEnv {
private:
    std::unique_ptr<minority> game;
    std::vector<int> history;
    
    // Environment parameters
    int num_players;
    int memory_size;
    int num_strategies;
    int equilibration_time;
    int max_episodes;
    int replace_agent_idx;
    long seed;
    int current_step;
    double rl_agent_score;
    int rl_agent_wins;
    
    std::vector<int> non_rl_agent_wins;  // Track wins for each non-RL agent
    
    // Calculate reward for RL agent
    double calculate_reward(int rl_action, int total_attendance);
    
public:
    MinorityGameEnv(int num_players = 101,
                   int memory_size = 3,
                   int num_strategies = 2,
                   int equilibration_time = 500,
                   int max_episodes = 10000,
                   int replace_agent_idx = 0,
                   long seed = -1);
    
    ~MinorityGameEnv();
    
    // Environment interface
    Observation reset();
    std::tuple<Observation, double, bool, EnvInfo> step(int action);
    
    // Getters
    int get_action_space_size() const { return 2; }  // Binary choice: 0 or 1
    int get_observation_space_size() const { return memory_size; }
    int get_current_step() const { return current_step; }
    double get_rl_agent_score() const { return rl_agent_score; }
    int get_rl_agent_wins() const { return rl_agent_wins; }
    
    std::vector<int> get_non_rl_agent_wins() const { return non_rl_agent_wins; }
    std::vector<double> get_non_rl_win_rates() const;
    double get_non_rl_avg_win_rate() const;
    double get_non_rl_std_win_rate() const;
    
    // Utility methods
    void render() const;
    Observation get_observation() const;
};

// Multi-agent RL environment
class MultiAgentMinorityGameEnv {
private:
    std::unique_ptr<minority> game;
    std::vector<int> history;
    std::vector<double> rl_agent_scores;
    std::vector<int> rl_agent_wins;
    
    // Environment parameters
    int num_players;
    int num_rl_agents;
    int memory_size;
    int num_strategies;
    int equilibration_time;
    int max_episodes;
    std::vector<int> rl_agent_indices;
    long seed;
    int current_step;
    
    std::vector<int> non_rl_agent_wins;  // Track wins for each non-RL agent
    
public:
    MultiAgentMinorityGameEnv(int num_players = 101,
                             int num_rl_agents = 2,
                             int memory_size = 3,
                             int num_strategies = 2,
                             int equilibration_time = 500,
                             int max_episodes = 10000,
                             long seed = -1);
    
    ~MultiAgentMinorityGameEnv();
    
    // Environment interface
    std::vector<Observation> reset();
    std::tuple<std::vector<Observation>, std::vector<double>, bool, EnvInfo> step(const std::vector<int>& actions);
    
    // Getters
    int get_action_space_size() const { return 2; }
    int get_observation_space_size() const { return memory_size; }
    int get_num_rl_agents() const { return num_rl_agents; }
    int get_current_step() const { return current_step; }
    const std::vector<double>& get_rl_agent_scores() const { return rl_agent_scores; }
    const std::vector<int>& get_rl_agent_wins() const { return rl_agent_wins; }
    
    std::vector<int> get_non_rl_agent_wins() const { return non_rl_agent_wins; }
    std::vector<double> get_non_rl_win_rates() const;
    double get_non_rl_avg_win_rate() const;
    double get_non_rl_std_win_rate() const;
    
    // Utility methods
    void render() const;
    Observation get_observation() const;
};

#endif // _MINORITY_GAME_ENV_H_
