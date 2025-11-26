/***************************************************************************
                          rl_agents.cpp  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#include "rl_agents.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <numeric>

// Base RLAgent Implementation

RLAgent::RLAgent(int obs_size, int action_size, unsigned int seed)
    : observation_space_size(obs_size), action_space_size(action_size), rng(seed) {}

// RandomAgent Implementation

RandomAgent::RandomAgent(int obs_size, int action_size, unsigned int seed)
    : RLAgent(obs_size, action_size, seed), action_dist(0, action_size - 1) {}

int RandomAgent::predict(const Observation& /* observation */, bool /* deterministic */) {
    return action_dist(rng);
}

void RandomAgent::learn(const Observation& /* obs */, int /* action */, double /* reward */, 
                       const Observation& /* next_obs */, bool /* done */) {
    // Random agent doesn't learn
}

void RandomAgent::save_model(const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "RandomAgent\n";
        file << "observation_space_size: " << observation_space_size << "\n";
        file << "action_space_size: " << action_space_size << "\n";
        file.close();
        std::cout << "Random agent model saved to " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
}

void RandomAgent::load_model(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        std::getline(file, line); // Agent type
        std::getline(file, line); // observation_space_size
        std::getline(file, line); // action_space_size
        file.close();
        std::cout << "Random agent model loaded from " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
}

// QLearningAgent Implementation

QLearningAgent::QLearningAgent(int obs_size, int action_size, double learning_rate,
                              double epsilon, double gamma, double epsilon_decay,
                              double epsilon_min, unsigned int seed)
    : RLAgent(obs_size, action_size, seed), learning_rate(learning_rate),
      epsilon(epsilon), gamma(gamma), epsilon_decay(epsilon_decay),
      epsilon_min(epsilon_min) {}

std::vector<int> QLearningAgent::obs_to_key(const Observation& observation) const {
    return observation.history;
}

void QLearningAgent::ensure_q_entry(const std::vector<int>& obs_key) {
    if (q_table.find(obs_key) == q_table.end()) {
        q_table[obs_key] = std::vector<double>(action_space_size, 0.0);
    }
}

int QLearningAgent::predict(const Observation& observation, bool deterministic) {
    std::vector<int> obs_key = obs_to_key(observation);
    ensure_q_entry(obs_key);
    
    if (!deterministic && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < epsilon) {
        // Explore: random action
        return std::uniform_int_distribution<int>(0, action_space_size - 1)(rng);
    } else {
        // Exploit: best action
        const auto& q_values = q_table[obs_key];
        return std::distance(q_values.begin(), 
                           std::max_element(q_values.begin(), q_values.end()));
    }
}

void QLearningAgent::learn(const Observation& obs, int action, double reward,
                          const Observation& next_obs, bool done) {
    std::vector<int> obs_key = obs_to_key(obs);
    std::vector<int> next_obs_key = obs_to_key(next_obs);
    
    ensure_q_entry(obs_key);
    ensure_q_entry(next_obs_key);
    
    // Q-learning update
    double current_q = q_table[obs_key][action];
    double next_max_q = done ? 0.0 : *std::max_element(q_table[next_obs_key].begin(),
                                                       q_table[next_obs_key].end());
    double target_q = reward + gamma * next_max_q;
    
    q_table[obs_key][action] += learning_rate * (target_q - current_q);
}

void QLearningAgent::save_model(const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "QLearningAgent\n";
        file << "observation_space_size: " << observation_space_size << "\n";
        file << "action_space_size: " << action_space_size << "\n";
        file << "learning_rate: " << learning_rate << "\n";
        file << "epsilon: " << epsilon << "\n";
        file << "gamma: " << gamma << "\n";
        file << "epsilon_decay: " << epsilon_decay << "\n";
        file << "epsilon_min: " << epsilon_min << "\n";
        file << "q_table_size: " << q_table.size() << "\n";
        
        // Save Q-table
        for (const auto& entry : q_table) {
            file << "state: ";
            for (int val : entry.first) {
                file << val << " ";
            }
            file << "| values: ";
            for (double val : entry.second) {
                file << val << " ";
            }
            file << "\n";
        }
        
        file.close();
        std::cout << "Q-Learning agent model saved to " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
}

void QLearningAgent::load_model(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        
        // Read header information
        std::getline(file, line); // Agent type
        std::getline(file, line); // observation_space_size
        std::getline(file, line); // action_space_size
        std::getline(file, line); // learning_rate
        std::getline(file, line); // epsilon
        std::getline(file, line); // gamma
        std::getline(file, line); // epsilon_decay
        std::getline(file, line); // epsilon_min
        std::getline(file, line); // q_table_size
        
        // Clear existing Q-table
        q_table.clear();
        
        // Read Q-table entries
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string token;
            
            // Read "state:"
            iss >> token;
            
            // Read state values
            std::vector<int> state;
            while (iss >> token && token != "|") {
                state.push_back(std::stoi(token));
            }
            
            // Read "values:"
            iss >> token;
            
            // Read Q-values
            std::vector<double> values;
            double value;
            while (iss >> value) {
                values.push_back(value);
            }
            
            q_table[state] = values;
        }
        
        file.close();
        std::cout << "Q-Learning agent model loaded from " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
}

// DQNAgent Implementation

DQNAgent::DQNAgent(int obs_size, int action_size, double learning_rate,
                  double epsilon, double gamma, double epsilon_decay,
                  double epsilon_min, int memory_capacity, int batch_size,
                  int target_update_frequency, unsigned int seed)
    : RLAgent(obs_size, action_size, seed), learning_rate(learning_rate),
      epsilon(epsilon), gamma(gamma), epsilon_decay(epsilon_decay),
      epsilon_min(epsilon_min), memory_capacity(memory_capacity),
      batch_size(batch_size), target_update_frequency(target_update_frequency),
      update_counter(0), buffer_index(0), buffer_full(false) {
    
    replay_buffer.reserve(memory_capacity);
}

std::vector<int> DQNAgent::obs_to_key(const Observation& observation) const {
    return observation.history;
}

void DQNAgent::ensure_network_entry(const std::vector<int>& obs_key) {
    if (main_network.find(obs_key) == main_network.end()) {
        // Initialize with small random values
        std::vector<double> values(action_space_size);
        std::uniform_real_distribution<double> init_dist(-0.1, 0.1);
        for (double& val : values) {
            val = init_dist(rng);
        }
        main_network[obs_key] = values;
        target_network[obs_key] = values;
    }
}

std::vector<double> DQNAgent::get_q_values(const std::vector<int>& obs_key, bool use_target) {
    ensure_network_entry(obs_key);
    return use_target ? target_network[obs_key] : main_network[obs_key];
}

void DQNAgent::update_target_network() {
    target_network = main_network;
}

int DQNAgent::predict(const Observation& observation, bool deterministic) {
    std::vector<int> obs_key = obs_to_key(observation);
    std::vector<double> q_values = get_q_values(obs_key);
    
    if (!deterministic && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < epsilon) {
        // Explore: random action
        return std::uniform_int_distribution<int>(0, action_space_size - 1)(rng);
    } else {
        // Exploit: best action
        return std::distance(q_values.begin(), 
                           std::max_element(q_values.begin(), q_values.end()));
    }
}

void DQNAgent::learn(const Observation& obs, int action, double reward,
                    const Observation& next_obs, bool done) {
    // Store experience in replay buffer
    std::vector<int> obs_key = obs_to_key(obs);
    std::vector<int> next_obs_key = obs_to_key(next_obs);
    
    if ((int)replay_buffer.size() < memory_capacity) {
        replay_buffer.emplace_back(obs_key, action, reward, next_obs_key, done);
    } else {
        replay_buffer[buffer_index] = Experience(obs_key, action, reward, next_obs_key, done);
        buffer_full = true;
    }
    
    buffer_index = (buffer_index + 1) % memory_capacity;
    
    // Perform experience replay if we have enough samples
    if ((int)get_replay_buffer_size() >= batch_size) {
        replay_experience();
    }
    
    // Update target network periodically
    update_counter++;
    if (update_counter % target_update_frequency == 0) {
        update_target_network();
    }
}

void DQNAgent::replay_experience() {
    int buffer_size = get_replay_buffer_size();
    
    // Sample random batch
    std::vector<int> indices(buffer_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    int actual_batch_size = std::min(batch_size, buffer_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        const Experience& exp = replay_buffer[indices[i]];
        
        // Get current Q-values
        std::vector<double> current_q_values = get_q_values(exp.state);
        
        // Calculate target Q-value
        double target_q;
        if (exp.done) {
            target_q = exp.reward;
        } else {
            std::vector<double> next_q_values = get_q_values(exp.next_state, true);
            double max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
            target_q = exp.reward + gamma * max_next_q;
        }
        
        // Update Q-value using gradient descent
        double current_q = current_q_values[exp.action];
        double td_error = target_q - current_q;
        current_q_values[exp.action] += learning_rate * td_error;
        
        // Update main network
        main_network[exp.state] = current_q_values;
    }
}

void DQNAgent::save_model(const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << "DQNAgent\n";
        file << "observation_space_size: " << observation_space_size << "\n";
        file << "action_space_size: " << action_space_size << "\n";
        file << "learning_rate: " << learning_rate << "\n";
        file << "epsilon: " << epsilon << "\n";
        file << "gamma: " << gamma << "\n";
        file << "epsilon_decay: " << epsilon_decay << "\n";
        file << "epsilon_min: " << epsilon_min << "\n";
        file << "memory_capacity: " << memory_capacity << "\n";
        file << "batch_size: " << batch_size << "\n";
        file << "target_update_frequency: " << target_update_frequency << "\n";
        file << "network_size: " << main_network.size() << "\n";
        
        // Save main network
        for (const auto& entry : main_network) {
            file << "state: ";
            for (int val : entry.first) {
                file << val << " ";
            }
            file << "| values: ";
            for (double val : entry.second) {
                file << val << " ";
            }
            file << "\n";
        }
        
        file.close();
        std::cout << "DQN agent model saved to " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
}

void DQNAgent::load_model(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        
        // Read header information
        for (int i = 0; i < 11; i++) {
            std::getline(file, line);
        }
        
        // Clear existing networks
        main_network.clear();
        target_network.clear();
        
        // Read network entries
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string token;
            
            // Read "state:"
            iss >> token;
            
            // Read state values
            std::vector<int> state;
            while (iss >> token && token != "|") {
                state.push_back(std::stoi(token));
            }
            
            // Read "values:"
            iss >> token;
            
            // Read Q-values
            std::vector<double> values;
            double value;
            while (iss >> value) {
                values.push_back(value);
            }
            
            main_network[state] = values;
            target_network[state] = values;
        }
        
        file.close();
        std::cout << "DQN agent model loaded from " << filepath << std::endl;
    } else {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
}

// Factory function implementation

std::unique_ptr<RLAgent> create_agent(const std::string& agent_type, 
                                     int obs_size, int action_size,
                                     const std::map<std::string, double>& params,
                                     unsigned int seed) {
    if (agent_type == "random") {
        return std::make_unique<RandomAgent>(obs_size, action_size, seed);
    } else if (agent_type == "qlearning") {
        double lr = params.count("learning_rate") ? params.at("learning_rate") : 0.1;
        double eps = params.count("epsilon") ? params.at("epsilon") : 0.1;
        double gamma = params.count("gamma") ? params.at("gamma") : 0.95;
        double eps_decay = params.count("epsilon_decay") ? params.at("epsilon_decay") : 0.995;
        double eps_min = params.count("epsilon_min") ? params.at("epsilon_min") : 0.01;
        
        return std::make_unique<QLearningAgent>(obs_size, action_size, lr, eps, 
                                               gamma, eps_decay, eps_min, seed);
    } else if (agent_type == "dqn") {
        double lr = params.count("learning_rate") ? params.at("learning_rate") : 0.001;
        double eps = params.count("epsilon") ? params.at("epsilon") : 1.0;
        double gamma = params.count("gamma") ? params.at("gamma") : 0.95;
        double eps_decay = params.count("epsilon_decay") ? params.at("epsilon_decay") : 0.995;
        double eps_min = params.count("epsilon_min") ? params.at("epsilon_min") : 0.01;
        int memory = params.count("memory_capacity") ? (int)params.at("memory_capacity") : 10000;
        int batch = params.count("batch_size") ? (int)params.at("batch_size") : 32;
        int target_freq = params.count("target_update_frequency") ? (int)params.at("target_update_frequency") : 100;
        
        return std::make_unique<DQNAgent>(obs_size, action_size, lr, eps, gamma,
                                         eps_decay, eps_min, memory, batch, 
                                         target_freq, seed);
    } else {
        throw std::invalid_argument("Unknown agent type: " + agent_type);
    }
}
