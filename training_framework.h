/***************************************************************************
                          training_framework.h  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#ifndef _TRAINING_FRAMEWORK_H_
#define _TRAINING_FRAMEWORK_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <chrono>
#include <fstream>
#include "minority_game_env.h"
#include "rl_agents.h"

// Training metrics for tracking performance
struct TrainingMetrics {
    std::vector<double> episode_rewards;
    std::vector<double> win_rates;
    std::vector<double> average_rewards;
    std::vector<int> episode_lengths;
    std::chrono::duration<double> training_time;
    
    TrainingMetrics() : training_time(0) {}
    
    void add_episode(double reward, double win_rate, int length) {
        episode_rewards.push_back(reward);
        win_rates.push_back(win_rate);
        episode_lengths.push_back(length);
        
        // Calculate running average reward
        if (episode_rewards.size() >= 100) {
            double sum = 0.0;
            for (int i = episode_rewards.size() - 100; i < (int)episode_rewards.size(); i++) {
                sum += episode_rewards[i];
            }
            average_rewards.push_back(sum / 100.0);
        } else {
            double sum = 0.0;
            for (double r : episode_rewards) {
                sum += r;
            }
            average_rewards.push_back(sum / episode_rewards.size());
        }
    }
    
    void save_to_file(const std::string& filepath) const;
    void load_from_file(const std::string& filepath);
    void print_summary() const;
};

// Configuration for training parameters
struct TrainingConfig {
    int episodes;
    int render_frequency;
    int save_frequency;
    bool save_model;
    bool verbose;
    std::string model_save_path;
    std::string metrics_save_path;
    
    // Environment parameters
    int num_players;
    int memory_size;
    int num_strategies;
    int equilibration_time;
    int max_episode_steps;
    long seed;
    
    // Agent parameters
    std::string agent_type;
    std::map<std::string, double> agent_params;
    
    TrainingConfig() 
        : episodes(1000), render_frequency(100), save_frequency(500),
          save_model(true), verbose(true), model_save_path("models/"),
          metrics_save_path("metrics/"), num_players(101), memory_size(3),
          num_strategies(2), equilibration_time(500), max_episode_steps(10000),
          seed(-1), agent_type("qlearning") {}
};

// Single agent trainer
class SingleAgentTrainer {
private:
    std::unique_ptr<MinorityGameEnv> env;
    std::unique_ptr<RLAgent> agent;
    TrainingConfig config;
    TrainingMetrics metrics;
    
    void create_directories();
    std::string generate_model_filename() const;
    std::string generate_metrics_filename() const;
    
public:
    SingleAgentTrainer(const TrainingConfig& config);
    ~SingleAgentTrainer() = default;
    
    // Training methods
    TrainingMetrics train();
    void evaluate(int num_episodes = 100);
    
    // Utility methods
    void save_model(const std::string& filepath = "");
    void load_model(const std::string& filepath);
    void save_metrics(const std::string& filepath = "");
    void load_metrics(const std::string& filepath);
    
    // Getters
    const TrainingMetrics& get_metrics() const { return metrics; }
    const TrainingConfig& get_config() const { return config; }
    RLAgent* get_agent() { return agent.get(); }
    MinorityGameEnv* get_environment() { return env.get(); }
};

// Multi-agent trainer configuration
struct MultiAgentTrainingConfig {
    int episodes;
    int render_frequency;
    int save_frequency;
    bool save_models;
    bool verbose;
    std::string model_save_path;
    std::string metrics_save_path;
    
    // Environment parameters
    int num_players;
    int num_rl_agents;
    int memory_size;
    int num_strategies;
    int equilibration_time;
    int max_episode_steps;
    long seed;
    
    // Agent parameters (can be different for each agent)
    std::vector<std::string> agent_types;
    std::vector<std::map<std::string, double>> agent_params;
    
    MultiAgentTrainingConfig()
        : episodes(500), render_frequency(50), save_frequency(250),
          save_models(true), verbose(true), model_save_path("models/"),
          metrics_save_path("metrics/"), num_players(101), num_rl_agents(2),
          memory_size(3), num_strategies(2), equilibration_time(500),
          max_episode_steps(10000), seed(-1) {
        
        // Default: 2 Q-learning agents
        agent_types = {"qlearning", "qlearning"};
        agent_params.resize(2);
    }
};

// Multi-agent training metrics
struct MultiAgentTrainingMetrics {
    std::vector<std::vector<double>> episode_rewards;  // [agent][episode]
    std::vector<std::vector<double>> win_rates;        // [agent][episode]
    std::vector<std::vector<double>> average_rewards;  // [agent][episode]
    std::vector<int> episode_lengths;
    std::chrono::duration<double> training_time;
    
    MultiAgentTrainingMetrics(int num_agents) : training_time(0) {
        episode_rewards.resize(num_agents);
        win_rates.resize(num_agents);
        average_rewards.resize(num_agents);
    }
    
    void add_episode(const std::vector<double>& rewards, 
                    const std::vector<double>& win_rates_vec, int length);
    void save_to_file(const std::string& filepath) const;
    void load_from_file(const std::string& filepath);
    void print_summary() const;
};

// Multi-agent trainer
class MultiAgentTrainer {
private:
    std::unique_ptr<MultiAgentMinorityGameEnv> env;
    std::vector<std::unique_ptr<RLAgent>> agents;
    MultiAgentTrainingConfig config;
    MultiAgentTrainingMetrics metrics;
    
    void create_directories();
    std::string generate_model_filename(int agent_idx) const;
    std::string generate_metrics_filename() const;
    
public:
    MultiAgentTrainer(const MultiAgentTrainingConfig& config);
    ~MultiAgentTrainer() = default;
    
    // Training methods
    MultiAgentTrainingMetrics train();
    void evaluate(int num_episodes = 100);
    
    // Utility methods
    void save_models(const std::string& base_path = "");
    void load_models(const std::vector<std::string>& filepaths);
    void save_metrics(const std::string& filepath = "");
    void load_metrics(const std::string& filepath);
    
    // Getters
    const MultiAgentTrainingMetrics& get_metrics() const { return metrics; }
    const MultiAgentTrainingConfig& get_config() const { return config; }
    const std::vector<std::unique_ptr<RLAgent>>& get_agents() const { return agents; }
    MultiAgentMinorityGameEnv* get_environment() { return env.get(); }
};

// Utility functions for agent comparison
struct ComparisonResult {
    std::string agent_name;
    double average_reward;
    double average_win_rate;
    double std_reward;
    double std_win_rate;
    
    ComparisonResult(const std::string& name) 
        : agent_name(name), average_reward(0.0), average_win_rate(0.0),
          std_reward(0.0), std_win_rate(0.0) {}
};

class AgentComparator {
private:
    TrainingConfig base_config;
    
public:
    AgentComparator(const TrainingConfig& config) : base_config(config) {}
    
    std::vector<ComparisonResult> compare_agents(
        const std::vector<std::string>& agent_types,
        const std::vector<std::map<std::string, double>>& agent_params,
        int num_episodes = 100);
    
    void print_comparison_results(const std::vector<ComparisonResult>& results);
    void save_comparison_results(const std::vector<ComparisonResult>& results,
                                const std::string& filepath);
};

// Training utilities
namespace TrainingUtils {
    // Create default configurations
    TrainingConfig create_default_single_config();
    MultiAgentTrainingConfig create_default_multi_config();
    
    // Parameter sweep utilities
    std::vector<std::map<std::string, double>> create_parameter_grid(
        const std::map<std::string, std::vector<double>>& param_ranges);
    
    // Statistical utilities
    double calculate_mean(const std::vector<double>& values);
    double calculate_std(const std::vector<double>& values);
    std::pair<double, double> calculate_confidence_interval(
        const std::vector<double>& values, double confidence = 0.95);
    
    // File utilities
    void ensure_directory_exists(const std::string& path);
    std::string get_timestamp_string();
}

#endif // _TRAINING_FRAMEWORK_H_
