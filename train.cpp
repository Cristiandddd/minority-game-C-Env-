/***************************************************************************
                          train.cpp  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "training_framework.h"
#include "rl_agents.h"
#include "minority_game_env.h"
#include "rnd.h"

// Function to display help information
void print_help() {
    std::cout << "Minority Game RL Training\n";
    std::cout << "Usage: train [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --agent TYPE          Agent type (random, qlearning, dqn) [default: qlearning]\n";
    std::cout << "  --episodes N          Number of training episodes [default: 1000]\n";
    std::cout << "  --players N           Number of players in the game [default: 101]\n";
    std::cout << "  --memory N            Memory size [default: 3]\n";
    std::cout << "  --lr RATE             Learning rate [default: 0.1]\n";
    std::cout << "  --epsilon EPS         Exploration rate [default: 0.1]\n";
    std::cout << "  --gamma GAMMA         Discount factor [default: 0.95]\n";
    std::cout << "  --seed N              Random seed [default: random]\n";
    std::cout << "  --multiagent N        Train N RL agents simultaneously\n";
    std::cout << "  --compare             Compare different agent types\n";
    std::cout << "  --evaluate FILE       Evaluate a saved model\n";
    std::cout << "  --sweep PLAYERS       Sweep mode: train across multiple player counts (e.g., \"101,301,501\")\n";
    std::cout << "  --sweep-episodes N    Episodes for each sweep configuration [default: 1000]\n";
    std::cout << "  --eval-episodes N     Evaluation episodes for sweep [default: 500]\n";
    std::cout << "  --output-csv FILE     CSV output file for sweep results [default: sweep_results.csv]\n";
    std::cout << "  --verbose             Enable verbose output [default: true]\n";
    std::cout << "  --help                Show this help message\n";
    std::cout << std::endl;
}

// Parse command line arguments
std::map<std::string, std::string> parse_arguments(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    
    // Set defaults
    args["agent"] = "qlearning";
    args["episodes"] = "1000";
    args["players"] = "101";
    args["memory"] = "3";
    args["lr"] = "0.1";
    args["epsilon"] = "0.1";
    args["gamma"] = "0.95";
    args["verbose"] = "true";
    args["sweep-episodes"] = "1000";
    args["eval-episodes"] = "500";
    args["output-csv"] = "sweep_results.csv";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            args["help"] = "true";
        } else if (arg == "--compare") {
            args["compare"] = "true";
        } else if (arg == "--verbose") {
            args["verbose"] = "true";
        } else if (i + 1 < argc) {
            std::string value = argv[i + 1];
            
            if (arg == "--agent") {
                args["agent"] = value;
                i++;
            } else if (arg == "--episodes") {
                args["episodes"] = value;
                i++;
            } else if (arg == "--players") {
                args["players"] = value;
                i++;
            } else if (arg == "--memory") {
                args["memory"] = value;
                i++;
            } else if (arg == "--lr") {
                args["lr"] = value;
                i++;
            } else if (arg == "--epsilon") {
                args["epsilon"] = value;
                i++;
            } else if (arg == "--gamma") {
                args["gamma"] = value;
                i++;
            } else if (arg == "--seed") {
                args["seed"] = value;
                i++;
            } else if (arg == "--multiagent") {
                args["multiagent"] = value;
                i++;
            } else if (arg == "--evaluate") {
                args["evaluate"] = value;
                i++;
            } else if (arg == "--sweep") {
                args["sweep"] = value;
                i++;
            } else if (arg == "--sweep-episodes") {
                args["sweep-episodes"] = value;
                i++;
            } else if (arg == "--eval-episodes") {
                args["eval-episodes"] = value;
                i++;
            } else if (arg == "--output-csv") {
                args["output-csv"] = value;
                i++;
            }
        }
    }
    
    return args;
}

// Train a single agent
void train_single_agent(const std::map<std::string, std::string>& args) {
    std::cout << "=== Single Agent Training ===" << std::endl;
    
    // Create training configuration
    TrainingConfig config = TrainingUtils::create_default_single_config();
    
    // Set parameters from arguments
    config.agent_type = args.at("agent");
    config.episodes = std::stoi(args.at("episodes"));
    config.num_players = std::stoi(args.at("players"));
    config.memory_size = std::stoi(args.at("memory"));
    config.verbose = (args.at("verbose") == "true");
    
    if (args.find("seed") != args.end()) {
        config.seed = std::stol(args.at("seed"));
    }
    
    // Set agent-specific parameters
    config.agent_params["learning_rate"] = std::stod(args.at("lr"));
    config.agent_params["epsilon"] = std::stod(args.at("epsilon"));
    config.agent_params["gamma"] = std::stod(args.at("gamma"));
    
    // Create and run trainer
    SingleAgentTrainer trainer(config);
    TrainingMetrics metrics = trainer.train();
    
    // Evaluate the trained agent
    std::cout << "\nEvaluating trained agent..." << std::endl;
    trainer.evaluate(100);
}

// Train multiple agents
void train_multi_agent(const std::map<std::string, std::string>& args) {
    std::cout << "=== Multi-Agent Training ===" << std::endl;
    
    int num_rl_agents = std::stoi(args.at("multiagent"));
    
    // Create training configuration
    MultiAgentTrainingConfig config = TrainingUtils::create_default_multi_config();
    
    // Set parameters from arguments
    config.episodes = std::stoi(args.at("episodes"));
    config.num_players = std::stoi(args.at("players"));
    config.num_rl_agents = num_rl_agents;
    config.memory_size = std::stoi(args.at("memory"));
    config.verbose = (args.at("verbose") == "true");
    
    if (args.find("seed") != args.end()) {
        config.seed = std::stol(args.at("seed"));
    }
    
    // Set up agents (all same type for simplicity)
    config.agent_types.clear();
    config.agent_params.clear();
    
    for (int i = 0; i < num_rl_agents; i++) {
        config.agent_types.push_back(args.at("agent"));
        
        std::map<std::string, double> agent_params;
        agent_params["learning_rate"] = std::stod(args.at("lr"));
        agent_params["epsilon"] = std::stod(args.at("epsilon"));
        agent_params["gamma"] = std::stod(args.at("gamma"));
        config.agent_params.push_back(agent_params);
    }
    
    // Create and run trainer
    MultiAgentTrainer trainer(config);
    MultiAgentTrainingMetrics metrics = trainer.train();
    
    // Evaluate the trained agents
    std::cout << "\nEvaluating trained agents..." << std::endl;
    trainer.evaluate(100);
}

// Compare different agent types
void compare_agents(const std::map<std::string, std::string>& args) {
    std::cout << "=== Agent Comparison ===" << std::endl;
    
    // Create base configuration
    TrainingConfig config = TrainingUtils::create_default_single_config();
    config.episodes = std::min(500, std::stoi(args.at("episodes"))); // Shorter for comparison
    config.num_players = std::stoi(args.at("players"));
    config.memory_size = std::stoi(args.at("memory"));
    config.verbose = false; // Reduce output for comparison
    
    if (args.find("seed") != args.end()) {
        config.seed = std::stol(args.at("seed"));
    }
    
    // Define agents to compare
    std::vector<std::string> agent_types = {"random", "qlearning", "dqn"};
    std::vector<std::map<std::string, double>> agent_params;
    
    // Random agent (no parameters)
    agent_params.push_back({});
    
    // Q-learning agent
    std::map<std::string, double> q_params;
    q_params["learning_rate"] = std::stod(args.at("lr"));
    q_params["epsilon"] = std::stod(args.at("epsilon"));
    q_params["gamma"] = std::stod(args.at("gamma"));
    agent_params.push_back(q_params);
    
    // DQN agent
    std::map<std::string, double> dqn_params;
    dqn_params["learning_rate"] = std::stod(args.at("lr")) * 0.1; // Lower LR for DQN
    dqn_params["epsilon"] = 1.0; // Start with high exploration
    dqn_params["gamma"] = std::stod(args.at("gamma"));
    dqn_params["memory_capacity"] = 5000;
    dqn_params["batch_size"] = 32;
    agent_params.push_back(dqn_params);
    
    // Run comparison
    AgentComparator comparator(config);
    std::vector<ComparisonResult> results = comparator.compare_agents(
        agent_types, agent_params, 100);
    
    comparator.print_comparison_results(results);
    
    // Save results
    std::string timestamp = TrainingUtils::get_timestamp_string();
    comparator.save_comparison_results(results, 
        "metrics/agent_comparison_" + timestamp + ".csv");
}

// Evaluate a saved model
void evaluate_model(const std::map<std::string, std::string>& args) {
    std::cout << "=== Model Evaluation ===" << std::endl;
    
    std::string model_path = args.at("evaluate");
    
    // Create environment
    MinorityGameEnv env(std::stoi(args.at("players")), 
                       std::stoi(args.at("memory")), 2, 500, 10000, 0, 
                       args.find("seed") != args.end() ? std::stol(args.at("seed")) : -1);
    
    // Create agent and load model
    std::unique_ptr<RLAgent> agent = create_agent(args.at("agent"), 
        env.get_observation_space_size(), env.get_action_space_size());
    
    try {
        agent->load_model(model_path);
        std::cout << "Model loaded successfully from: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return;
    }
    
    // Evaluate
    std::vector<double> eval_rewards;
    std::vector<double> eval_win_rates;
    int num_episodes = 100;
    
    for (int episode = 0; episode < num_episodes; episode++) {
        Observation obs = env.reset();
        double total_reward = 0.0;
        
        for (int step = 0; step < 10000; step++) {
            int action = agent->predict(obs, true); // Deterministic
            auto [next_obs, reward, terminated, info] = env.step(action);
            
            obs = next_obs;
            total_reward += reward;
            
            if (terminated) {
                eval_rewards.push_back(total_reward);
                eval_win_rates.push_back(info.win_rate);
                break;
            }
        }
        
        if ((episode + 1) % 20 == 0) {
            std::cout << "Evaluated " << (episode + 1) << "/" << num_episodes 
                      << " episodes..." << std::endl;
        }
    }
    
    double avg_reward = TrainingUtils::calculate_mean(eval_rewards);
    double std_reward = TrainingUtils::calculate_std(eval_rewards);
    double avg_win_rate = TrainingUtils::calculate_mean(eval_win_rates);
    double std_win_rate = TrainingUtils::calculate_std(eval_win_rates);
    
    std::cout << "\n=== Evaluation Results ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Episodes: " << num_episodes << std::endl;
    std::cout << "Average Reward: " << std::fixed << std::setprecision(3) 
              << avg_reward << " ± " << std_reward << std::endl;
    std::cout << "Average Win Rate: " << std::fixed << std::setprecision(1) 
              << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
    std::cout << "=========================\n" << std::endl;
}

// Parse comma-separated player counts
std::vector<int> parse_player_counts(const std::string& players_str) {
    std::vector<int> player_counts;
    std::stringstream ss(players_str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        try {
            int count = std::stoi(item);
            if (count > 0) {
                player_counts.push_back(count);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid player count '" << item << "', skipping." << std::endl;
        }
    }
    
    return player_counts;
}

// Generate model filename for sweep
std::string generate_sweep_model_filename(const std::string& agent_type, int memory, int players) {
    return agent_type + "_agent_m" + std::to_string(memory) + "_n" + std::to_string(players) + ".model";
}

// Sweep training across multiple player configurations
void run_sweep_training(const std::map<std::string, std::string>& args) {
    std::cout << "=== Parameter Sweep Training ===" << std::endl;
    
    // Parse player counts
    std::vector<int> player_counts = parse_player_counts(args.at("sweep"));
    if (player_counts.empty()) {
        std::cerr << "Error: No valid player counts provided for sweep." << std::endl;
        return;
    }
    
    // Get parameters
    std::string agent_type = args.at("agent");
    int memory_size = std::stoi(args.at("memory"));
    int sweep_episodes = std::stoi(args.at("sweep-episodes"));
    int eval_episodes = std::stoi(args.at("eval-episodes"));
    std::string output_csv = args.at("output-csv");
    bool verbose = (args.at("verbose") == "true");
    
    double learning_rate = std::stod(args.at("lr"));
    double epsilon = std::stod(args.at("epsilon"));
    double gamma = std::stod(args.at("gamma"));
    
    long seed = -1;
    if (args.find("seed") != args.end()) {
        seed = std::stol(args.at("seed"));
    }
    
    std::cout << "Sweep Configuration:" << std::endl;
    std::cout << "  Agent Type: " << agent_type << std::endl;
    std::cout << "  Memory Size: " << memory_size << " (fixed)" << std::endl;
    std::cout << "  Player Counts: ";
    for (size_t i = 0; i < player_counts.size(); i++) {
        std::cout << player_counts[i];
        if (i < player_counts.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  Training Episodes: " << sweep_episodes << std::endl;
    std::cout << "  Evaluation Episodes: " << eval_episodes << std::endl;
    std::cout << "  Output CSV: " << output_csv << std::endl;
    std::cout << std::endl;
    
    // Prepare CSV file
    std::ofstream csv_file(output_csv);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open output CSV file: " << output_csv << std::endl;
        return;
    }
    
    csv_file << "agent_type,memory_size,num_players,learning_rate,epsilon,gamma,";
    csv_file << "average_reward,std_reward,win_rate,std_win_rate,";
    csv_file << "non_rl_avg_win_rate,non_rl_std_win_rate,model_filename" << std::endl;
    
    // Run sweep for each player count
    for (int num_players : player_counts) {
        std::cout << "\n--- Training with " << num_players << " players ---" << std::endl;
        
        // Create training configuration
        TrainingConfig config = TrainingUtils::create_default_single_config();
        config.agent_type = agent_type;
        config.episodes = sweep_episodes;
        config.num_players = num_players;
        config.memory_size = memory_size;
        config.verbose = verbose;
        config.seed = seed;
        
        // Set agent parameters
        config.agent_params["learning_rate"] = learning_rate;
        config.agent_params["epsilon"] = epsilon;
        config.agent_params["gamma"] = gamma;
        
        // Generate model filename
        std::string model_filename = generate_sweep_model_filename(agent_type, memory_size, num_players);
        config.model_save_path = "sweep_models/";
        
        try {
            // Create and train agent
            SingleAgentTrainer trainer(config);
            std::cout << "Training agent..." << std::endl;
            TrainingMetrics training_metrics = trainer.train();
            
            // Save the trained model
            std::cout << "Saving model to: " << model_filename << std::endl;
            trainer.save_model("sweep_models/" + model_filename);
            
            // Evaluate the trained agent
            std::cout << "Evaluating agent over " << eval_episodes << " episodes..." << std::endl;
            
            // Create evaluation environment
            MinorityGameEnv eval_env(num_players, memory_size, 2, 500, 10000, 0, seed);
            RLAgent* agent = trainer.get_agent();
            
            std::vector<double> eval_rewards;
            std::vector<double> eval_win_rates;
            std::vector<double> non_rl_std_win_rates_vec;
            
            for (int episode = 0; episode < eval_episodes; episode++) {
                Observation obs = eval_env.reset();
                double total_reward = 0.0;
                
                for (int step = 0; step < 10000; step++) {
                    int action = agent->predict(obs, true); // Deterministic evaluation
                    auto [next_obs, reward, terminated, info] = eval_env.step(action);
                    
                    obs = next_obs;
                    total_reward += reward;
                    
                    if (terminated) {
                        eval_rewards.push_back(total_reward);
                        eval_win_rates.push_back(info.win_rate);
                        non_rl_std_win_rates_vec.push_back(eval_env.get_non_rl_std_win_rate());
                        break;
                    }
                }
                
                if (verbose && (episode + 1) % 100 == 0) {
                    std::cout << "  Evaluated " << (episode + 1) << "/" << eval_episodes 
                              << " episodes..." << std::endl;
                }
            }
            
            // Calculate statistics
            double avg_reward = TrainingUtils::calculate_mean(eval_rewards);
            double std_reward = TrainingUtils::calculate_std(eval_rewards);
            double avg_win_rate = TrainingUtils::calculate_mean(eval_win_rates);
            double std_win_rate = TrainingUtils::calculate_std(eval_win_rates);
            
            double non_rl_avg_wr = eval_env.get_non_rl_avg_win_rate();
            double non_rl_std_wr = TrainingUtils::calculate_mean(non_rl_std_win_rates_vec);
            
            // Print results
            std::cout << "Results for " << num_players << " players:" << std::endl;
            std::cout << "  Average Reward: " << std::fixed << std::setprecision(4) 
                      << avg_reward << " ± " << std_reward << std::endl;
            std::cout << "  Win Rate: " << std::fixed << std::setprecision(2) 
                      << avg_win_rate * 100 << "% ± " << std_win_rate * 100 << "%" << std::endl;
            std::cout << "  Non-RL Avg Win Rate: " << std::fixed << std::setprecision(2) 
                      << non_rl_avg_wr * 100 << "%" << std::endl;
            std::cout << "  Non-RL Std Win Rate: " << std::fixed << std::setprecision(2) 
                      << non_rl_std_wr * 100 << "%" << std::endl;
            std::cout << "  Model saved as: " << model_filename << std::endl;
            
            csv_file << agent_type << "," << memory_size << "," << num_players << ",";
            csv_file << learning_rate << "," << epsilon << "," << gamma << ",";
            csv_file << std::fixed << std::setprecision(6) << avg_reward << "," << std_reward << ",";
            csv_file << avg_win_rate << "," << std_win_rate << ",";
            csv_file << non_rl_avg_wr << "," << non_rl_std_wr << ",";
            csv_file << model_filename << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error training with " << num_players << " players: " << e.what() << std::endl;
            
            // Write error entry to CSV
            csv_file << agent_type << "," << memory_size << "," << num_players << ",";
            csv_file << learning_rate << "," << epsilon << "," << gamma << ",";
            csv_file << "ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" << std::endl;
        }
    }
    
    csv_file.close();
    
    std::cout << "\n=== Sweep Training Complete ===" << std::endl;
    std::cout << "Results saved to: " << output_csv << std::endl;
    std::cout << "Models saved in: sweep_models/" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Minority Game RL Training System" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Parse command line arguments
    std::map<std::string, std::string> args = parse_arguments(argc, argv);
    
    // Show help if requested
    if (args.find("help") != args.end()) {
        print_help();
        return 0;
    }
    
    // Initialize random number generator
    if (args.find("seed") != args.end()) {
        long seed = std::stol(args.at("seed"));
        RNDInit(seed);
        std::cout << "Using seed: " << seed << std::endl;
    } else {
        long seed = RNDInit(-1); // Use time-based seed
        std::cout << "Using random seed: " << seed << std::endl;
    }
    
    try {
        // Determine what to do based on arguments
        if (args.find("evaluate") != args.end()) {
            evaluate_model(args);
        } else if (args.find("compare") != args.end()) {
            compare_agents(args);
        } else if (args.find("multiagent") != args.end()) {
            train_multi_agent(args);
        } else if (args.find("sweep") != args.end()) {
            run_sweep_training(args);
        } else {
            train_single_agent(args);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nProgram completed successfully!" << std::endl;
    return 0;
}
