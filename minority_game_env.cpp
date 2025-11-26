/***************************************************************************
                          minority_game_env.cpp  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#include "minority_game_env.h"
#include "rnd.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

// Single Agent Environment Implementation

MinorityGameEnv::MinorityGameEnv(int num_players, int memory_size, int num_strategies,
                                int equilibration_time, int max_episodes, 
                                int replace_agent_idx, long seed)
    : num_players(num_players), memory_size(memory_size), num_strategies(num_strategies),
      equilibration_time(equilibration_time), max_episodes(max_episodes),
      replace_agent_idx(replace_agent_idx), seed(seed),
      current_step(0), rl_agent_score(0.0), rl_agent_wins(0) {
    
    if (seed != -1) {
        RNDInit(seed);
    }
    
    non_rl_agent_wins.resize(num_players - 1, 0);  // All agents except the RL agent
}

MinorityGameEnv::~MinorityGameEnv() = default;

Observation MinorityGameEnv::reset() {
    // Create new game instance
    minority_options opts;
    opts.number_of_players = num_players;
    opts.memory = memory_size;
    opts.number_of_strategies = num_strategies;
    opts.teq = equilibration_time;
    opts.seed = seed;
    
    game = std::make_unique<minority>(opts);
    
    // Reset state
    history.clear();
    current_step = 0;
    rl_agent_score = 0.0;
    rl_agent_wins = 0;
    
    std::fill(non_rl_agent_wins.begin(), non_rl_agent_wins.end(), 0);
    
    // Initialize history with random outcomes
    for (int i = 0; i < memory_size; i++) {
        history.push_back(RNDInteger(1));  // 0 or 1
    }
    
    return get_observation();
}

std::tuple<Observation, double, bool, EnvInfo> MinorityGameEnv::step(int action) {
    if (!game) {
        throw std::runtime_error("Environment not initialized. Call reset() first.");
    }
    
    // Convert RL action (0/1) to game bet (-1/+1)
    int rl_bet = (action == 1) ? 1 : -1;
    
    // Get current memory state for traditional agents
    unsigned long P = 1UL << memory_size;
    unsigned long mu = 0;
    if (!history.empty()) {
        // Convert history to memory state
        for (int i = 0; i < std::min(memory_size, (int)history.size()); i++) {
            int idx = history.size() - memory_size + i;
            if (idx >= 0) {
                mu += history[idx] * (1UL << i);
            }
        }
        mu = mu % P;
    }
    
    unsigned long mu_naive = RNDInteger(P - 1);  // Random state for naive agents
    
    // Collect bets from all agents
    int total_attendance = 0;
    std::vector<int> agent_bets;
    
    std::vector<agent> players = game->GetPlayers();
    for (int i = 0; i < (int)players.size(); i++) {
        int bet;
        if (i == replace_agent_idx) {
            // Use RL agent's action
            bet = rl_bet;
        } else {
            // Use traditional agent's strategy
            bet = players[i].Bet(mu, mu_naive);
        }
        
        agent_bets.push_back(bet);
        total_attendance += bet;
    }
    
    // Determine winning side (minority)
    int winning_side;
    if (total_attendance > 0) {
        // More players chose +1, so minority is -1 (action 0)
        winning_side = 0;
    } else if (total_attendance < 0) {
        // More players chose -1, so minority is +1 (action 1)
        winning_side = 1;
    } else {
        // Tie: equal number of +1 and -1, random winner
        winning_side = RNDInteger(1);
    }
    
    double reward;
    if (action == winning_side) {
        reward = 1.0;  // Won by being in minority
        rl_agent_wins++;
    } else {
        reward = -1.0;  // Lost by being in majority
    }
    
    // Additional penalty based on attendance (closer to 0 is better)
    double attendance_penalty = std::abs(total_attendance) / (double)num_players;
    reward -= attendance_penalty * 0.1;  // Small penalty for extreme imbalances
    
    rl_agent_score += reward;
    
    int non_rl_idx = 0;
    for (int i = 0; i < (int)players.size(); i++) {
        if (i != replace_agent_idx) {
            // Check if this non-RL agent won
            int agent_action = (agent_bets[i] == 1) ? 1 : 0;
            if (agent_action == winning_side) {
                non_rl_agent_wins[non_rl_idx]++;
            }
            non_rl_idx++;
        }
    }
    
    // Update traditional agent scores
    for (int i = 0; i < (int)players.size(); i++) {
        if (i != replace_agent_idx) {
            unsigned long memory_state = players[i].Naive() ? mu_naive : mu;
            players[i].UpdateScore(memory_state, total_attendance);
        }
    }
    
    // Update history with the winning_side calculated above
    history.push_back(winning_side);
    
    // Update step counter
    current_step++;
    
    // Check if episode is done
    bool terminated = (current_step >= max_episodes);
    
    // Get next observation
    Observation observation = get_observation();
    
    // Prepare info
    EnvInfo info;
    info.step = current_step;
    info.total_attendance = total_attendance;
    info.winning_side = winning_side;
    info.rl_agent_score = rl_agent_score;
    info.rl_agent_wins = rl_agent_wins;
    info.win_rate = (current_step > 0) ? (double)rl_agent_wins / current_step : 0.0;
    info.agent_bets = agent_bets;
    info.memory_state = mu;
    
    info.non_rl_agent_wins = non_rl_agent_wins;
    info.non_rl_win_rates = get_non_rl_win_rates();
    
    return std::make_tuple(observation, reward, terminated, info);
}

// This eliminates the double calculation of winning_side and potential double counting

std::vector<double> MinorityGameEnv::get_non_rl_win_rates() const {
    std::vector<double> win_rates;
    if (current_step > 0) {
        for (int wins : non_rl_agent_wins) {
            win_rates.push_back((double)wins / current_step);
        }
    } else {
        win_rates.resize(non_rl_agent_wins.size(), 0.0);
    }
    return win_rates;
}

double MinorityGameEnv::get_non_rl_avg_win_rate() const {
    if (non_rl_agent_wins.empty() || current_step == 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int wins : non_rl_agent_wins) {
        sum += (double)wins / current_step;
    }
    return sum / non_rl_agent_wins.size();
}

double MinorityGameEnv::get_non_rl_std_win_rate() const {
    if (non_rl_agent_wins.empty() || current_step == 0) {
        return 0.0;
    }
    
    double avg = get_non_rl_avg_win_rate();
    double sum_sq_diff = 0.0;
    
    for (int wins : non_rl_agent_wins) {
        double win_rate = (double)wins / current_step;
        double diff = win_rate - avg;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / non_rl_agent_wins.size());
}

Observation MinorityGameEnv::get_observation() const {
    Observation obs(memory_size);
    obs.update(history, memory_size);
    return obs;
}

void MinorityGameEnv::render() const {
    std::cout << "Step: " << current_step << std::endl;
    std::cout << "History (last 10): ";
    int start = std::max(0, (int)history.size() - 10);
    for (int i = start; i < (int)history.size(); i++) {
        std::cout << history[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "RL Agent Score: " << rl_agent_score << std::endl;
    std::cout << "Win Rate: " << ((current_step > 0) ? (double)rl_agent_wins / current_step * 100 : 0) << "%" << std::endl;
    std::cout << "Non-RL Avg Win Rate: " << get_non_rl_avg_win_rate() * 100 << "%" << std::endl;
    std::cout << "Non-RL Std Win Rate: " << get_non_rl_std_win_rate() * 100 << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// Multi-Agent Environment Implementation

MultiAgentMinorityGameEnv::MultiAgentMinorityGameEnv(int num_players, int num_rl_agents,
                                                   int memory_size, int num_strategies,
                                                   int equilibration_time, int max_episodes,
                                                   long seed)
    : num_players(num_players), num_rl_agents(num_rl_agents), memory_size(memory_size),
      num_strategies(num_strategies), equilibration_time(equilibration_time),
      max_episodes(max_episodes), seed(seed), current_step(0) {
    
    if (seed != -1) {
        RNDInit(seed);
    }
    
    // RL agent indices (replace first num_rl_agents traditional agents)
    rl_agent_indices.clear();
    for (int i = 0; i < num_rl_agents; i++) {
        rl_agent_indices.push_back(i);
    }
    
    // Initialize score vectors
    rl_agent_scores.resize(num_rl_agents, 0.0);
    rl_agent_wins.resize(num_rl_agents, 0);
    
    non_rl_agent_wins.resize(num_players - num_rl_agents, 0);
}

MultiAgentMinorityGameEnv::~MultiAgentMinorityGameEnv() = default;

std::vector<Observation> MultiAgentMinorityGameEnv::reset() {
    // Create new game instance
    minority_options opts;
    opts.number_of_players = num_players;
    opts.memory = memory_size;
    opts.number_of_strategies = num_strategies;
    opts.teq = equilibration_time;
    opts.seed = seed;
    
    game = std::make_unique<minority>(opts);
    
    // Reset state
    history.clear();
    current_step = 0;
    std::fill(rl_agent_scores.begin(), rl_agent_scores.end(), 0.0);
    std::fill(rl_agent_wins.begin(), rl_agent_wins.end(), 0);
    
    std::fill(non_rl_agent_wins.begin(), non_rl_agent_wins.end(), 0);
    
    // Initialize history with random outcomes
    for (int i = 0; i < memory_size; i++) {
        history.push_back(RNDInteger(1));
    }
    
    // Get initial observations for all RL agents (same observation for all)
    Observation obs = get_observation();
    std::vector<Observation> observations;
    for (int i = 0; i < num_rl_agents; i++) {
        observations.push_back(obs);
    }
    
    return observations;
}

std::tuple<std::vector<Observation>, std::vector<double>, bool, EnvInfo> 
MultiAgentMinorityGameEnv::step(const std::vector<int>& actions) {
    if ((int)actions.size() != num_rl_agents) {
        throw std::invalid_argument("Expected " + std::to_string(num_rl_agents) + 
                                  " actions, got " + std::to_string(actions.size()));
    }
    
    // Convert actions to bets
    std::vector<int> rl_bets;
    for (int action : actions) {
        rl_bets.push_back((action == 1) ? 1 : -1);
    }
    
    // Get memory state
    unsigned long P = 1UL << memory_size;
    unsigned long mu = 0;
    if (!history.empty()) {
        for (int i = 0; i < std::min(memory_size, (int)history.size()); i++) {
            int idx = history.size() - memory_size + i;
            if (idx >= 0) {
                mu += history[idx] * (1UL << i);
            }
        }
        mu = mu % P;
    }
    
    unsigned long mu_naive = RNDInteger(P - 1);
    
    // Collect all bets
    int total_attendance = 0;
    std::vector<int> agent_bets;
    
    std::vector<agent> players = game->GetPlayers();
    for (int i = 0; i < (int)players.size(); i++) {
        int bet;
        auto it = std::find(rl_agent_indices.begin(), rl_agent_indices.end(), i);
        if (it != rl_agent_indices.end()) {
            // Use RL agent's action
            int rl_idx = std::distance(rl_agent_indices.begin(), it);
            bet = rl_bets[rl_idx];
        } else {
            // Use traditional agent's strategy
            bet = players[i].Bet(mu, mu_naive);
        }
        
        agent_bets.push_back(bet);
        total_attendance += bet;
    }
    
    // Determine winning side (minority)
    int winning_side;
    if (total_attendance > 0) {
        // More players chose +1, so minority is -1 (action 0)
        winning_side = 0;
    } else if (total_attendance < 0) {
        // More players chose -1, so minority is +1 (action 1)
        winning_side = 1;
    } else {
        // Tie: equal number of +1 and -1, random winner
        winning_side = RNDInteger(1);
    }
    
    int non_rl_idx = 0;
    for (int i = 0; i < (int)players.size(); i++) {
        if (std::find(rl_agent_indices.begin(), rl_agent_indices.end(), i) == rl_agent_indices.end()) {
            // This is a non-RL agent
            int agent_action = (agent_bets[i] == 1) ? 1 : 0;
            if (agent_action == winning_side) {
                non_rl_agent_wins[non_rl_idx]++;
            }
            non_rl_idx++;
        }
    }
    
    // Calculate rewards for RL agents using the SAME winning_side
    std::vector<double> rewards;
    for (int i = 0; i < num_rl_agents; i++) {
        double reward;
        if (actions[i] == winning_side) {
            reward = 1.0;
            rl_agent_wins[i]++;
        } else {
            reward = -1.0;
        }
        
        // Add attendance penalty
        double attendance_penalty = std::abs(total_attendance) / (double)num_players;
        reward -= attendance_penalty * 0.1;
        
        rl_agent_scores[i] += reward;
        rewards.push_back(reward);
    }
    
    // Update traditional agents
    for (int i = 0; i < (int)players.size(); i++) {
        if (std::find(rl_agent_indices.begin(), rl_agent_indices.end(), i) == rl_agent_indices.end()) {
            unsigned long memory_state = players[i].Naive() ? mu_naive : mu;
            players[i].UpdateScore(memory_state, total_attendance);
        }
    }
    
    // Update history and step
    history.push_back(winning_side);
    current_step++;
    
    // Check termination
    bool terminated = (current_step >= max_episodes);
    
    // Get observations for all RL agents (same observation for all)
    Observation obs = get_observation();
    std::vector<Observation> observations;
    for (int i = 0; i < num_rl_agents; i++) {
        observations.push_back(obs);
    }
    
    // Info
    EnvInfo info;
    info.step = current_step;
    info.total_attendance = total_attendance;
    info.winning_side = winning_side;
    info.agent_bets = agent_bets;
    info.memory_state = mu;
    
    info.non_rl_agent_wins = non_rl_agent_wins;
    info.non_rl_win_rates = get_non_rl_win_rates();  // This now properly initializes the vector
    
    return std::make_tuple(observations, rewards, terminated, info);
}

std::vector<double> MultiAgentMinorityGameEnv::get_non_rl_win_rates() const {
    std::vector<double> win_rates;
    if (current_step > 0) {
        for (int wins : non_rl_agent_wins) {
            win_rates.push_back((double)wins / current_step);
        }
    } else {
        win_rates.resize(non_rl_agent_wins.size(), 0.0);
    }
    return win_rates;
}

double MultiAgentMinorityGameEnv::get_non_rl_avg_win_rate() const {
    if (non_rl_agent_wins.empty() || current_step == 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int wins : non_rl_agent_wins) {
        sum += (double)wins / current_step;
    }
    return sum / non_rl_agent_wins.size();
}

double MultiAgentMinorityGameEnv::get_non_rl_std_win_rate() const {
    if (non_rl_agent_wins.empty() || current_step == 0) {
        return 0.0;
    }
    
    double avg = get_non_rl_avg_win_rate();
    double sum_sq_diff = 0.0;
    
    for (int wins : non_rl_agent_wins) {
        double win_rate = (double)wins / current_step;
        double diff = win_rate - avg;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / non_rl_agent_wins.size());
}

Observation MultiAgentMinorityGameEnv::get_observation() const {
    Observation obs(memory_size);
    obs.update(history, memory_size);
    return obs;
}

void MultiAgentMinorityGameEnv::render() const {
    std::cout << "Step: " << current_step << std::endl;
    std::cout << "History (last 10): ";
    int start = std::max(0, (int)history.size() - 10);
    for (int i = start; i < (int)history.size(); i++) {
        std::cout << history[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "RL Agent Scores: ";
    for (double score : rl_agent_scores) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    std::cout << "Win Rates: ";
    for (int wins : rl_agent_wins) {
        double rate = (current_step > 0) ? (double)wins / current_step * 100 : 0;
        std::cout << rate << "% ";
    }
    std::cout << std::endl;
    std::cout << "Non-RL Avg Win Rate: " << get_non_rl_avg_win_rate() * 100 << "%" << std::endl;
    std::cout << "Non-RL Std Win Rate: " << get_non_rl_std_win_rate() * 100 << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}
