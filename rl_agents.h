/***************************************************************************
                          rl_agents.h  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#ifndef _RL_AGENTS_H_
#define _RL_AGENTS_H_

#include <vector>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <fstream>
#include "minority_game_env.h"

// Base class for RL agents
class RLAgent {
protected:
    int observation_space_size;
    int action_space_size;
    std::mt19937 rng;
    
public:
    RLAgent(int obs_size, int action_size, unsigned int seed = 0);
    virtual ~RLAgent() = default;
    
    // Pure virtual methods that must be implemented by derived classes
    virtual int predict(const Observation& observation, bool deterministic = false) = 0;
    virtual void learn(const Observation& obs, int action, double reward, 
                      const Observation& next_obs, bool done) = 0;
    
    // Model persistence
    virtual void save_model(const std::string& filepath) = 0;
    virtual void load_model(const std::string& filepath) = 0;
    
    // Getters
    int get_observation_space_size() const { return observation_space_size; }
    int get_action_space_size() const { return action_space_size; }
};

// Random agent for baseline comparison
class RandomAgent : public RLAgent {
private:
    std::uniform_int_distribution<int> action_dist;
    
public:
    RandomAgent(int obs_size, int action_size, unsigned int seed = 0);
    
    int predict(const Observation& observation, bool deterministic = false) override;
    void learn(const Observation& obs, int action, double reward, 
              const Observation& next_obs, bool done) override;
    
    void save_model(const std::string& filepath) override;
    void load_model(const std::string& filepath) override;
};

// Simple Q-Learning agent
class QLearningAgent : public RLAgent {
private:
    double learning_rate;
    double epsilon;
    double gamma;
    double epsilon_decay;
    double epsilon_min;
    
    // Q-table: maps observation to action values
    std::map<std::vector<int>, std::vector<double>> q_table;
    
    // Helper methods
    std::vector<int> obs_to_key(const Observation& observation) const;
    void ensure_q_entry(const std::vector<int>& obs_key);
    
public:
    QLearningAgent(int obs_size, int action_size, 
                  double learning_rate = 0.1, double epsilon = 0.1, 
                  double gamma = 0.95, double epsilon_decay = 0.995,
                  double epsilon_min = 0.01, unsigned int seed = 0);
    
    int predict(const Observation& observation, bool deterministic = false) override;
    void learn(const Observation& obs, int action, double reward, 
              const Observation& next_obs, bool done) override;
    
    void save_model(const std::string& filepath) override;
    void load_model(const std::string& filepath) override;
    
    // Getters for hyperparameters
    double get_learning_rate() const { return learning_rate; }
    double get_epsilon() const { return epsilon; }
    double get_gamma() const { return gamma; }
    size_t get_q_table_size() const { return q_table.size(); }
    
    // Setters for hyperparameters
    void set_epsilon(double new_epsilon) { epsilon = std::max(epsilon_min, new_epsilon); }
    void decay_epsilon() { epsilon = std::max(epsilon_min, epsilon * epsilon_decay); }
};

// Deep Q-Network agent (simplified version using tabular approximation)
class DQNAgent : public RLAgent {
private:
    double learning_rate;
    double epsilon;
    double gamma;
    double epsilon_decay;
    double epsilon_min;
    int memory_capacity;
    int batch_size;
    int target_update_frequency;
    int update_counter;
    
    // Experience replay buffer
    struct Experience {
        std::vector<int> state;
        int action;
        double reward;
        std::vector<int> next_state;
        bool done;
        
        Experience(const std::vector<int>& s, int a, double r, 
                  const std::vector<int>& ns, bool d)
            : state(s), action(a), reward(r), next_state(ns), done(d) {}
    };
    
    std::vector<Experience> replay_buffer;
    int buffer_index;
    bool buffer_full;
    
    // Neural network approximation using lookup table
    std::map<std::vector<int>, std::vector<double>> main_network;
    std::map<std::vector<int>, std::vector<double>> target_network;
    
    // Helper methods
    std::vector<int> obs_to_key(const Observation& observation) const;
    void ensure_network_entry(const std::vector<int>& obs_key);
    std::vector<double> get_q_values(const std::vector<int>& obs_key, bool use_target = false);
    void update_target_network();
    void replay_experience();
    
public:
    DQNAgent(int obs_size, int action_size,
            double learning_rate = 0.001, double epsilon = 1.0,
            double gamma = 0.95, double epsilon_decay = 0.995,
            double epsilon_min = 0.01, int memory_capacity = 10000,
            int batch_size = 32, int target_update_frequency = 100,
            unsigned int seed = 0);
    
    int predict(const Observation& observation, bool deterministic = false) override;
    void learn(const Observation& obs, int action, double reward, 
              const Observation& next_obs, bool done) override;
    
    void save_model(const std::string& filepath) override;
    void load_model(const std::string& filepath) override;
    
    // Getters
    double get_learning_rate() const { return learning_rate; }
    double get_epsilon() const { return epsilon; }
    double get_gamma() const { return gamma; }
    size_t get_replay_buffer_size() const { return buffer_full ? memory_capacity : buffer_index; }
    size_t get_network_size() const { return main_network.size(); }
    
    // Setters
    void set_epsilon(double new_epsilon) { epsilon = std::max(epsilon_min, new_epsilon); }
    void decay_epsilon() { epsilon = std::max(epsilon_min, epsilon * epsilon_decay); }
};

// Factory function to create agents
std::unique_ptr<RLAgent> create_agent(const std::string& agent_type, 
                                     int obs_size, int action_size,
                                     const std::map<std::string, double>& params = {},
                                     unsigned int seed = 0);

#endif // _RL_AGENTS_H_
