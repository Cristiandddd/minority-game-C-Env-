/***************************************************************************
                          training_framework.cpp  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#include "training_framework.h"
#include "rnd.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>

// TrainingMetrics Implementation

void TrainingMetrics::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "episode,reward,win_rate,average_reward,episode_length\n";
        for (size_t i = 0; i < episode_rewards.size(); i++) {
            file << i << "," << episode_rewards[i] << "," << win_rates[i] << ",";
            if (i < average_rewards.size()) {
                file << average_rewards[i];
            }
            file << ",";
            if (i < episode_lengths.size()) {
                file << episode_lengths[i];
            }
            file << "\n";
        }
        file << "training_time_seconds," << training_time.count() << "\n";
        file.close();
        std::cout << "Training metrics saved to " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
}

void TrainingMetrics::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        std::getline(file, line); // Skip header
        
        episode_rewards.clear();
        win_rates.clear();
        average_rewards.clear();
        episode_lengths.clear();
        
        while (std::getline(file, line)) {
            if (line.find("training_time_seconds") != std::string::npos) {
                size_t comma_pos = line.find(',');
                if (comma_pos != std::string::npos) {
                    double seconds = std::stod(line.substr(comma_pos + 1));
                    training_time = std::chrono::duration<double>(seconds);
                }
                break;
            }
            
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 5) {
                episode_rewards.push_back(std::stod(tokens[1]));
                win_rates.push_back(std::stod(tokens[2]));
                if (!tokens[3].empty()) {
                    average_rewards.push_back(std::stod(tokens[3]));
                }
                if (!tokens[4].empty()) {
                    episode_lengths.push_back(std::stoi(tokens[4]));
                }
            }
        }
        
        file.close();
        std::cout << "Training metrics loaded from " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
}

void TrainingMetrics::print_summary() const {
    if (episode_rewards.empty()) {
        std::cout << "No training data available." << std::endl;
        return;
    }
    
    double avg_reward = TrainingUtils::calculate_mean(episode_rewards);
    double std_reward = TrainingUtils::calculate_std(episode_rewards);
    double avg_win_rate = TrainingUtils::calculate_mean(win_rates);
    double std_win_rate = TrainingUtils::calculate_std(win_rates);
    
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "Episodes: " << episode_rewards.size() << std::endl;
    std::cout << "Training Time: " << training_time.count() << " seconds" << std::endl;
    std::cout << "Average Reward: " << std::fixed << std::setprecision(3) 
              << avg_reward << " ± " << std_reward << std::endl;
    std::cout << "Average Win Rate: " << std::fixed << std::setprecision(1) 
              << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
    
    if (!average_rewards.empty()) {
        std::cout << "Final 100-episode Average: " << std::fixed << std::setprecision(3)
                  << average_rewards.back() << std::endl;
    }
    std::cout << "========================\n" << std::endl;
}

// SingleAgentTrainer Implementation

SingleAgentTrainer::SingleAgentTrainer(const TrainingConfig& config) : config(config) {
    // Create environment
    env = std::make_unique<MinorityGameEnv>(
        config.num_players, config.memory_size, config.num_strategies,
        config.equilibration_time, config.max_episode_steps, 0, config.seed
    );
    
    // Create agent
    agent = create_agent(config.agent_type, env->get_observation_space_size(),
                        env->get_action_space_size(), config.agent_params, config.seed);
    
    create_directories();
}

void SingleAgentTrainer::create_directories() {
    TrainingUtils::ensure_directory_exists(config.model_save_path);
    TrainingUtils::ensure_directory_exists(config.metrics_save_path);
}

std::string SingleAgentTrainer::generate_model_filename() const {
    return config.model_save_path + config.agent_type + "_agent_" + 
           TrainingUtils::get_timestamp_string() + ".model";
}

std::string SingleAgentTrainer::generate_metrics_filename() const {
    return config.metrics_save_path + config.agent_type + "_metrics_" + 
           TrainingUtils::get_timestamp_string() + ".csv";
}

TrainingMetrics SingleAgentTrainer::train() {
    std::cout << "Starting single agent training..." << std::endl;
    std::cout << "Agent: " << config.agent_type << std::endl;
    std::cout << "Episodes: " << config.episodes << std::endl;
    std::cout << "Environment: " << config.num_players << " players, " 
              << config.memory_size << " memory" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < config.episodes; episode++) {
        Observation obs = env->reset();
        double total_reward = 0.0;
        
        EnvInfo final_info;
        
        for (int step = 0; step < config.max_episode_steps; step++) {
            // Select action
            int action = agent->predict(obs);
            
            // Take step
            auto [next_obs, reward, terminated, info] = env->step(action);
            
            // Learn
            agent->learn(obs, action, reward, next_obs, terminated);
            
            obs = next_obs;
            total_reward += reward;
            
            if (terminated) {
                final_info = info;
                break;
            }
        }
        
        // Record metrics
        metrics.add_episode(total_reward, final_info.win_rate, final_info.step);
        
        // Decay epsilon for exploration
        if (config.agent_type == "qlearning" || config.agent_type == "dqn") {
            if (auto* q_agent = dynamic_cast<QLearningAgent*>(agent.get())) {
                q_agent->decay_epsilon();
            } else if (auto* dqn_agent = dynamic_cast<DQNAgent*>(agent.get())) {
                dqn_agent->decay_epsilon();
            }
        }
        
        // Render occasionally
        if (config.verbose && episode % config.render_frequency == 0) {
            std::cout << "Episode " << episode << ": Reward = " << std::fixed 
                      << std::setprecision(2) << total_reward 
                      << ", Win Rate = " << std::setprecision(1) 
                      << final_info.win_rate * 100 << "%" << std::endl;
        }
        
        // Save model periodically
        if (config.save_model && episode % config.save_frequency == 0 && episode > 0) {
            save_model();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.training_time = end_time - start_time;
    
    // Final save
    if (config.save_model) {
        save_model();
    }
    save_metrics();
    
    std::cout << "Training completed!" << std::endl;
    metrics.print_summary();
    
    return metrics;
}

void SingleAgentTrainer::evaluate(int num_episodes) {
    std::cout << "Evaluating agent for " << num_episodes << " episodes..." << std::endl;
    
    std::vector<double> eval_rewards;
    std::vector<double> eval_win_rates;
    
    for (int episode = 0; episode < num_episodes; episode++) {
        Observation obs = env->reset();
        double total_reward = 0.0;
        
        EnvInfo final_info;
        
        for (int step = 0; step < config.max_episode_steps; step++) {
            int action = agent->predict(obs, true); // Deterministic
            auto [next_obs, reward, terminated, info] = env->step(action);
            
            obs = next_obs;
            total_reward += reward;
            
            if (terminated) {
                final_info = info;
                break;
            }
        }
        
        eval_rewards.push_back(total_reward);
        eval_win_rates.push_back(final_info.win_rate);
    }
    
    double avg_reward = TrainingUtils::calculate_mean(eval_rewards);
    double std_reward = TrainingUtils::calculate_std(eval_rewards);
    double avg_win_rate = TrainingUtils::calculate_mean(eval_win_rates);
    double std_win_rate = TrainingUtils::calculate_std(eval_win_rates);
    
    std::cout << "\n=== Evaluation Results ===" << std::endl;
    std::cout << "Episodes: " << num_episodes << std::endl;
    std::cout << "Average Reward: " << std::fixed << std::setprecision(3) 
              << avg_reward << " ± " << std_reward << std::endl;
    std::cout << "Average Win Rate: " << std::fixed << std::setprecision(1) 
              << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
    std::cout << "=========================\n" << std::endl;
}

void SingleAgentTrainer::save_model(const std::string& filepath) {
    std::string path = filepath.empty() ? generate_model_filename() : filepath;
    agent->save_model(path);
}

void SingleAgentTrainer::load_model(const std::string& filepath) {
    agent->load_model(filepath);
}

void SingleAgentTrainer::save_metrics(const std::string& filepath) {
    std::string path = filepath.empty() ? generate_metrics_filename() : filepath;
    metrics.save_to_file(path);
}

void SingleAgentTrainer::load_metrics(const std::string& filepath) {
    metrics.load_from_file(filepath);
}

// MultiAgentTrainingMetrics Implementation

void MultiAgentTrainingMetrics::add_episode(const std::vector<double>& rewards,
                                           const std::vector<double>& win_rates_vec, 
                                           int length) {
    for (size_t i = 0; i < rewards.size(); i++) {
        episode_rewards[i].push_back(rewards[i]);
        win_rates[i].push_back(win_rates_vec[i]);
        
        // Calculate running average reward
        if (episode_rewards[i].size() >= 100) {
            double sum = 0.0;
            for (int j = episode_rewards[i].size() - 100; j < (int)episode_rewards[i].size(); j++) {
                sum += episode_rewards[i][j];
            }
            average_rewards[i].push_back(sum / 100.0);
        } else {
            double sum = 0.0;
            for (double r : episode_rewards[i]) {
                sum += r;
            }
            average_rewards[i].push_back(sum / episode_rewards[i].size());
        }
    }
    episode_lengths.push_back(length);
}

void MultiAgentTrainingMetrics::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "episode";
        for (size_t i = 0; i < episode_rewards.size(); i++) {
            file << ",agent" << i << "_reward,agent" << i << "_win_rate,agent" << i << "_avg_reward";
        }
        file << ",episode_length\n";
        
        size_t max_episodes = episode_rewards.empty() ? 0 : episode_rewards[0].size();
        for (size_t ep = 0; ep < max_episodes; ep++) {
            file << ep;
            for (size_t i = 0; i < episode_rewards.size(); i++) {
                file << "," << episode_rewards[i][ep] << "," << win_rates[i][ep];
                if (ep < average_rewards[i].size()) {
                    file << "," << average_rewards[i][ep];
                } else {
                    file << ",";
                }
            }
            if (ep < episode_lengths.size()) {
                file << "," << episode_lengths[ep];
            }
            file << "\n";
        }
        
        file << "training_time_seconds," << training_time.count() << "\n";
        file.close();
        std::cout << "Multi-agent training metrics saved to " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
}

void MultiAgentTrainingMetrics::print_summary() const {
    if (episode_rewards.empty() || episode_rewards[0].empty()) {
        std::cout << "No training data available." << std::endl;
        return;
    }
    
    std::cout << "\n=== Multi-Agent Training Summary ===" << std::endl;
    std::cout << "Episodes: " << episode_rewards[0].size() << std::endl;
    std::cout << "Training Time: " << training_time.count() << " seconds" << std::endl;
    
    for (size_t i = 0; i < episode_rewards.size(); i++) {
        double avg_reward = TrainingUtils::calculate_mean(episode_rewards[i]);
        double std_reward = TrainingUtils::calculate_std(episode_rewards[i]);
        double avg_win_rate = TrainingUtils::calculate_mean(win_rates[i]);
        double std_win_rate = TrainingUtils::calculate_std(win_rates[i]);
        
        std::cout << "Agent " << i << ":" << std::endl;
        std::cout << "  Average Reward: " << std::fixed << std::setprecision(3) 
                  << avg_reward << " ± " << std_reward << std::endl;
        std::cout << "  Average Win Rate: " << std::fixed << std::setprecision(1) 
                  << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
    }
    std::cout << "===================================\n" << std::endl;
}

// MultiAgentTrainer Implementation

MultiAgentTrainer::MultiAgentTrainer(const MultiAgentTrainingConfig& config) 
    : config(config), metrics(config.num_rl_agents) {
    
    // Create environment
    env = std::make_unique<MultiAgentMinorityGameEnv>(
        config.num_players, config.num_rl_agents, config.memory_size,
        config.num_strategies, config.equilibration_time, 
        config.max_episode_steps, config.seed
    );
    
    // Create agents
    for (int i = 0; i < config.num_rl_agents; i++) {
        std::string agent_type = (i < (int)config.agent_types.size()) ? 
                                config.agent_types[i] : "qlearning";
        std::map<std::string, double> agent_params = (i < (int)config.agent_params.size()) ?
                                                    config.agent_params[i] : std::map<std::string, double>{};
        
        agents.push_back(create_agent(agent_type, env->get_observation_space_size(),
                                     env->get_action_space_size(), agent_params, config.seed + i));
    }
    
    create_directories();
}

void MultiAgentTrainer::create_directories() {
    TrainingUtils::ensure_directory_exists(config.model_save_path);
    TrainingUtils::ensure_directory_exists(config.metrics_save_path);
}

std::string MultiAgentTrainer::generate_model_filename(int agent_idx) const {
    std::string agent_type = (agent_idx < (int)config.agent_types.size()) ? 
                            config.agent_types[agent_idx] : "qlearning";
    return config.model_save_path + "multiagent_" + agent_type + "_agent" + 
           std::to_string(agent_idx) + "_" + TrainingUtils::get_timestamp_string() + ".model";
}

std::string MultiAgentTrainer::generate_metrics_filename() const {
    return config.metrics_save_path + "multiagent_metrics_" + 
           TrainingUtils::get_timestamp_string() + ".csv";
}

MultiAgentTrainingMetrics MultiAgentTrainer::train() {
    std::cout << "Starting multi-agent training..." << std::endl;
    std::cout << "Agents: " << config.num_rl_agents << std::endl;
    std::cout << "Episodes: " << config.episodes << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < config.episodes; episode++) {
        std::vector<Observation> observations = env->reset();
        std::vector<double> total_rewards(config.num_rl_agents, 0.0);
        
        std::vector<EnvInfo> final_infos(config.num_rl_agents);
        
        for (int step = 0; step < config.max_episode_steps; step++) {
            // Select actions for all agents
            std::vector<int> actions;
            for (int i = 0; i < config.num_rl_agents; i++) {
                actions.push_back(agents[i]->predict(observations[i]));
            }
            
            // Take step
            auto [next_observations, rewards, terminated, info] = env->step(actions);
            
            // Learn for all agents
            for (int i = 0; i < config.num_rl_agents; i++) {
                agents[i]->learn(observations[i], actions[i], rewards[i], 
                               next_observations[i], terminated);
                total_rewards[i] += rewards[i];
            }
            
            observations = next_observations;
            
            if (terminated) {
                final_infos.push_back(info);
                break;
            }
        }
        
        // Get win rates from environment (assuming they're stored in info somehow)
        std::vector<double> win_rates(config.num_rl_agents);
        for (int i = 0; i < config.num_rl_agents; i++) {
            win_rates[i] = final_infos[i].win_rate;
        }
        
        // Record metrics
        metrics.add_episode(total_rewards, win_rates, env->get_current_step());
        
        // Decay epsilon for exploration
        for (int i = 0; i < config.num_rl_agents; i++) {
            if (auto* q_agent = dynamic_cast<QLearningAgent*>(agents[i].get())) {
                q_agent->decay_epsilon();
            } else if (auto* dqn_agent = dynamic_cast<DQNAgent*>(agents[i].get())) {
                dqn_agent->decay_epsilon();
            }
        }
        
        // Render occasionally
        if (config.verbose && episode % config.render_frequency == 0) {
            std::cout << "Episode " << episode << ":" << std::endl;
            for (int i = 0; i < config.num_rl_agents; i++) {
                std::cout << "  Agent " << i << ": Reward = " << std::fixed 
                          << std::setprecision(2) << total_rewards[i] 
                          << ", Win Rate = " << std::setprecision(1) 
                          << final_infos[i].win_rate * 100 << "%" << std::endl;
            }
        }
        
        // Save models periodically
        if (config.save_models && episode % config.save_frequency == 0 && episode > 0) {
            save_models();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.training_time = end_time - start_time;
    
    // Final save
    if (config.save_models) {
        save_models();
    }
    save_metrics();
    
    std::cout << "Multi-agent training completed!" << std::endl;
    metrics.print_summary();
    
    return metrics;
}

void MultiAgentTrainer::save_models(const std::string& base_path) {
    for (int i = 0; i < config.num_rl_agents; i++) {
        std::string path = base_path.empty() ? generate_model_filename(i) : 
                          base_path + "_agent" + std::to_string(i) + ".model";
        agents[i]->save_model(path);
    }
}

void MultiAgentTrainer::save_metrics(const std::string& filepath) {
    std::string path = filepath.empty() ? generate_metrics_filename() : filepath;
    metrics.save_to_file(path);
}

// TrainingUtils Implementation

namespace TrainingUtils {
    TrainingConfig create_default_single_config() {
        return TrainingConfig();
    }
    
    MultiAgentTrainingConfig create_default_multi_config() {
        return MultiAgentTrainingConfig();
    }
    
    double calculate_mean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    
    double calculate_std(const std::vector<double>& values) {
        if (values.size() <= 1) return 0.0;
        
        double mean = calculate_mean(values);
        double sum_sq_diff = 0.0;
        for (double val : values) {
            sum_sq_diff += (val - mean) * (val - mean);
        }
        return std::sqrt(sum_sq_diff / (values.size() - 1));
    }
    
    void ensure_directory_exists(const std::string& path) {
        std::filesystem::create_directories(path);
    }
    
    std::string get_timestamp_string() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
}

// MultiAgentTrainer::evaluate implementation
void MultiAgentTrainer::evaluate(int num_episodes) {
    std::cout << "Evaluating multi-agent system for " << num_episodes << " episodes..." << std::endl;
    
    std::vector<std::vector<double>> eval_rewards(config.num_rl_agents);
    std::vector<std::vector<double>> eval_win_rates(config.num_rl_agents);
    
    for (int episode = 0; episode < num_episodes; episode++) {
        std::vector<Observation> observations = env->reset();
        std::vector<double> total_rewards(config.num_rl_agents, 0.0);
        
        std::vector<EnvInfo> final_infos(config.num_rl_agents);
        
        for (int step = 0; step < config.max_episode_steps; step++) {
            // Select actions deterministically
            std::vector<int> actions;
            for (int i = 0; i < config.num_rl_agents; i++) {
                actions.push_back(agents[i]->predict(observations[i], true));
            }
            
            // Take step
            auto [next_observations, rewards, terminated, info] = env->step(actions);
            
            for (int i = 0; i < config.num_rl_agents; i++) {
                total_rewards[i] += rewards[i];
            }
            
            observations = next_observations;
            
            if (terminated) {
                final_infos.push_back(info);
                break;
            }
        }
        
        for (int i = 0; i < config.num_rl_agents; i++) {
            eval_rewards[i].push_back(total_rewards[i]);
            eval_win_rates[i].push_back(final_infos[i].win_rate);
        }
        
        if ((episode + 1) % 20 == 0) {
            std::cout << "Evaluated " << (episode + 1) << "/" << num_episodes 
                      << " episodes..." << std::endl;
        }
    }
    
    std::cout << "\n=== Multi-Agent Evaluation Results ===" << std::endl;
    for (int i = 0; i < config.num_rl_agents; i++) {
        double avg_reward = TrainingUtils::calculate_mean(eval_rewards[i]);
        double std_reward = TrainingUtils::calculate_std(eval_rewards[i]);
        double avg_win_rate = TrainingUtils::calculate_mean(eval_win_rates[i]);
        double std_win_rate = TrainingUtils::calculate_std(eval_win_rates[i]);
        
        std::cout << "Agent " << i << ":" << std::endl;
        std::cout << "  Average Reward: " << std::fixed << std::setprecision(3) 
                  << avg_reward << " ± " << std_reward << std::endl;
        std::cout << "  Average Win Rate: " << std::fixed << std::setprecision(1) 
                  << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
    }
    std::cout << "=====================================\n" << std::endl;
}

// AgentComparator::compare_agents implementation
std::vector<ComparisonResult> AgentComparator::compare_agents(
    const std::vector<std::string>& agent_types,
    const std::vector<std::map<std::string, double>>& agent_params,
    int num_episodes) {
    
    std::vector<ComparisonResult> results;
    
    for (size_t i = 0; i < agent_types.size(); i++) {
        std::cout << "Testing agent: " << agent_types[i] << std::endl;
        
        // Create configuration for this agent
        TrainingConfig agent_config = base_config;
        agent_config.agent_type = agent_types[i];
        agent_config.agent_params = agent_params[i];
        agent_config.episodes = num_episodes;
        agent_config.verbose = false;
        
        // Train and evaluate
        SingleAgentTrainer trainer(agent_config);
        TrainingMetrics metrics = trainer.train();
        
        ComparisonResult result(agent_types[i]);
        result.average_reward = TrainingUtils::calculate_mean(metrics.episode_rewards);
        result.std_reward = TrainingUtils::calculate_std(metrics.episode_rewards);
        result.average_win_rate = TrainingUtils::calculate_mean(metrics.win_rates);
        result.std_win_rate = TrainingUtils::calculate_std(metrics.win_rates);
        
        results.push_back(result);
    }
    
    return results;
}

// AgentComparator::print_comparison_results implementation
void AgentComparator::print_comparison_results(const std::vector<ComparisonResult>& results) {
    std::cout << "\n=== Agent Comparison Results ===" << std::endl;
    std::cout << std::setw(15) << "Agent" 
              << std::setw(15) << "Avg Reward" 
              << std::setw(15) << "Std Reward"
              << std::setw(15) << "Win Rate"
              << std::setw(15) << "Std Win Rate" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.agent_name
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.average_reward
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.std_reward
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.average_win_rate * 100 << "%"
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.std_win_rate * 100 << "%" << std::endl;
    }
    std::cout << "================================\n" << std::endl;
}

// AgentComparator::save_comparison_results implementation
void AgentComparator::save_comparison_results(const std::vector<ComparisonResult>& results,
                                            const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "agent,avg_reward,std_reward,avg_win_rate,std_win_rate\n";
        for (const auto& result : results) {
            file << result.agent_name << ","
                 << result.average_reward << ","
                 << result.std_reward << ","
                 << result.average_win_rate << ","
                 << result.std_win_rate << "\n";
        }
        file.close();
        std::cout << "Comparison results saved to " << filepath << std::endl;
    } else {
        std::cerr << "Warning: Could not save comparison results to " << filepath << std::endl;
    }
}