/***************************************************************************
                          model_persistence.cpp  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#include "model_persistence.h"
#include "training_framework.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ctime>

// ModelMetadata Implementation

void ModelMetadata::save_to_stream(std::ofstream& file) const {
    file << "METADATA_START\n";
    file << "agent_type: " << agent_type << "\n";
    file << "creation_timestamp: " << creation_timestamp << "\n";
    file << "version: " << version << "\n";
    file << "observation_space_size: " << observation_space_size << "\n";
    file << "action_space_size: " << action_space_size << "\n";
    file << "final_average_reward: " << final_average_reward << "\n";
    file << "final_win_rate: " << final_win_rate << "\n";
    file << "training_episodes: " << training_episodes << "\n";
    file << "training_time_seconds: " << training_time_seconds << "\n";
    
    file << "hyperparameters_count: " << hyperparameters.size() << "\n";
    for (const auto& param : hyperparameters) {
        file << "hp_" << param.first << ": " << param.second << "\n";
    }
    
    file << "training_info_count: " << training_info.size() << "\n";
    for (const auto& info : training_info) {
        file << "ti_" << info.first << ": " << info.second << "\n";
    }
    
    file << "METADATA_END\n";
}

void ModelMetadata::load_from_stream(std::ifstream& file) {
    std::string line;
    
    // Find metadata start
    while (std::getline(file, line)) {
        if (line == "METADATA_START") break;
    }
    
    hyperparameters.clear();
    training_info.clear();
    
    while (std::getline(file, line) && line != "METADATA_END") {
        size_t colon_pos = line.find(": ");
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 2);
            
            if (key == "agent_type") agent_type = value;
            else if (key == "creation_timestamp") creation_timestamp = value;
            else if (key == "version") version = value;
            else if (key == "observation_space_size") observation_space_size = std::stoi(value);
            else if (key == "action_space_size") action_space_size = std::stoi(value);
            else if (key == "final_average_reward") final_average_reward = std::stod(value);
            else if (key == "final_win_rate") final_win_rate = std::stod(value);
            else if (key == "training_episodes") training_episodes = std::stoi(value);
            else if (key == "training_time_seconds") training_time_seconds = std::stod(value);
            else if (key.substr(0, 3) == "hp_") {
                hyperparameters[key.substr(3)] = std::stod(value);
            } else if (key.substr(0, 3) == "ti_") {
                training_info[key.substr(3)] = value;
            }
        }
    }
}

void ModelMetadata::print_info() const {
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Agent Type: " << agent_type << std::endl;
    std::cout << "Version: " << version << std::endl;
    std::cout << "Created: " << creation_timestamp << std::endl;
    std::cout << "Observation Space: " << observation_space_size << std::endl;
    std::cout << "Action Space: " << action_space_size << std::endl;
    std::cout << "Training Episodes: " << training_episodes << std::endl;
    std::cout << "Training Time: " << std::fixed << std::setprecision(2) 
              << training_time_seconds << " seconds" << std::endl;
    std::cout << "Final Average Reward: " << std::fixed << std::setprecision(3) 
              << final_average_reward << std::endl;
    std::cout << "Final Win Rate: " << std::fixed << std::setprecision(1) 
              << final_win_rate * 100 << "%" << std::endl;
    
    if (!hyperparameters.empty()) {
        std::cout << "Hyperparameters:" << std::endl;
        for (const auto& param : hyperparameters) {
            std::cout << "  " << param.first << ": " << param.second << std::endl;
        }
    }
    
    if (!training_info.empty()) {
        std::cout << "Training Info:" << std::endl;
        for (const auto& info : training_info) {
            std::cout << "  " << info.first << ": " << info.second << std::endl;
        }
    }
    std::cout << "========================\n" << std::endl;
}

// ModelPersistenceManager Implementation

ModelPersistenceManager::ModelPersistenceManager(const std::string& base_path) 
    : base_path(base_path) {
    models_directory = base_path + "models/";
    metadata_directory = base_path + "metadata/";
    ensure_directories_exist();
}

void ModelPersistenceManager::ensure_directories_exist() {
    std::filesystem::create_directories(models_directory);
    std::filesystem::create_directories(metadata_directory);
}

std::string ModelPersistenceManager::generate_model_filename(const std::string& agent_type,
                                                           const std::string& suffix) const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    std::string filename = agent_type + "_model_" + ss.str();
    if (!suffix.empty()) {
        filename += "_" + suffix;
    }
    filename += ".model";
    
    return models_directory + filename;
}

std::string ModelPersistenceManager::generate_metadata_filename(const std::string& model_filename) const {
    std::filesystem::path model_path(model_filename);
    std::string base_name = model_path.stem().string();
    return metadata_directory + base_name + ".meta";
}

void ModelPersistenceManager::save_model_with_metadata(RLAgent* agent, 
                                                      const ModelMetadata& metadata,
                                                      const std::string& custom_name) {
    std::string model_path;
    if (custom_name.empty()) {
        model_path = generate_model_filename(metadata.agent_type);
    } else {
        model_path = models_directory + ModelPersistenceUtils::sanitize_filename(custom_name) + ".model";
    }
    
    // Save the model
    agent->save_model(model_path);
    
    // Save metadata
    std::string metadata_path = generate_metadata_filename(model_path);
    std::ofstream meta_file(metadata_path);
    if (meta_file.is_open()) {
        metadata.save_to_stream(meta_file);
        meta_file.close();
        std::cout << "Model and metadata saved successfully:" << std::endl;
        std::cout << "  Model: " << model_path << std::endl;
        std::cout << "  Metadata: " << metadata_path << std::endl;
    } else {
        std::cerr << "Warning: Could not save metadata to " << metadata_path << std::endl;
    }
}

std::unique_ptr<RLAgent> ModelPersistenceManager::load_model_with_metadata(
    const std::string& model_path, ModelMetadata& metadata) {
    
    // Load metadata first
    std::string metadata_path = generate_metadata_filename(model_path);
    std::ifstream meta_file(metadata_path);
    if (meta_file.is_open()) {
        metadata.load_from_stream(meta_file);
        meta_file.close();
    } else {
        std::cerr << "Warning: Could not load metadata from " << metadata_path << std::endl;
        // Try to infer basic metadata from model file
        metadata.agent_type = "unknown";
    }
    
    // Create and load agent
    std::unique_ptr<RLAgent> agent = create_agent(
        metadata.agent_type, 
        metadata.observation_space_size,
        metadata.action_space_size,
        metadata.hyperparameters
    );
    
    agent->load_model(model_path);
    
    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    metadata.print_info();
    
    return agent;
}

std::vector<std::string> ModelPersistenceManager::list_available_models() const {
    std::vector<std::string> models;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(models_directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".model") {
                models.push_back(entry.path().string());
            }
        }
        
        // Sort by modification time (newest first)
        std::sort(models.begin(), models.end(), [](const std::string& a, const std::string& b) {
            return std::filesystem::last_write_time(a) > std::filesystem::last_write_time(b);
        });
        
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error listing models: " << e.what() << std::endl;
    }
    
    return models;
}

std::string ModelPersistenceManager::find_best_model(const std::string& agent_type,
                                                    const std::string& metric) const {
    std::vector<std::string> models = list_available_models();
    std::string best_model;
    double best_value = -std::numeric_limits<double>::infinity();
    
    for (const std::string& model_path : models) {
        try {
            ModelMetadata metadata;
            std::string metadata_path = generate_metadata_filename(model_path);
            std::ifstream meta_file(metadata_path);
            
            if (meta_file.is_open()) {
                metadata.load_from_stream(meta_file);
                meta_file.close();
                
                if (agent_type.empty() || metadata.agent_type == agent_type) {
                    double value = 0.0;
                    if (metric == "final_average_reward") {
                        value = metadata.final_average_reward;
                    } else if (metric == "final_win_rate") {
                        value = metadata.final_win_rate;
                    }
                    
                    if (value > best_value) {
                        best_value = value;
                        best_model = model_path;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading metadata for " << model_path << ": " << e.what() << std::endl;
        }
    }
    
    return best_model;
}

// TrainingCheckpointManager Implementation

TrainingCheckpointManager::TrainingCheckpointManager(const std::string& checkpoint_dir,
                                                   int frequency, int max_checkpoints)
    : checkpoint_directory(checkpoint_dir), checkpoint_frequency(frequency),
      max_checkpoints(max_checkpoints) {
    std::filesystem::create_directories(checkpoint_directory);
}

void TrainingCheckpointManager::save_checkpoint(RLAgent* agent, const TrainingMetrics& metrics,
                                               int episode, const std::string& experiment_name) {
    std::string exp_dir = checkpoint_directory;
    if (!experiment_name.empty()) {
        exp_dir += experiment_name + "/";
        std::filesystem::create_directories(exp_dir);
    }
    
    std::string checkpoint_path = exp_dir + "checkpoint_episode_" + std::to_string(episode) + ".ckpt";
    
    // Save agent model
    agent->save_model(checkpoint_path);
    
    // Save training metrics
    std::string metrics_path = exp_dir + "metrics_episode_" + std::to_string(episode) + ".csv";
    metrics.save_to_file(metrics_path);
    
    // Save checkpoint info
    std::string info_path = exp_dir + "checkpoint_info_" + std::to_string(episode) + ".txt";
    std::ofstream info_file(info_path);
    if (info_file.is_open()) {
        info_file << "episode: " << episode << "\n";
        info_file << "timestamp: " << std::time(nullptr) << "\n";
        info_file << "total_episodes: " << metrics.episode_rewards.size() << "\n";
        if (!metrics.episode_rewards.empty()) {
            info_file << "latest_reward: " << metrics.episode_rewards.back() << "\n";
            info_file << "latest_win_rate: " << metrics.win_rates.back() << "\n";
        }
        info_file.close();
    }
    
    std::cout << "Checkpoint saved at episode " << episode << ": " << checkpoint_path << std::endl;
    
    // Cleanup old checkpoints
    cleanup_old_checkpoints(experiment_name);
}

bool TrainingCheckpointManager::load_latest_checkpoint(std::unique_ptr<RLAgent>& agent,
                                                      TrainingMetrics& metrics, int& start_episode,
                                                      const std::string& experiment_name) {
    std::string exp_dir = checkpoint_directory;
    if (!experiment_name.empty()) {
        exp_dir += experiment_name + "/";
    }
    
    // Find latest checkpoint
    std::vector<std::string> checkpoints;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(exp_dir)) {
            if (entry.is_regular_file() && 
                entry.path().filename().string().find("checkpoint_episode_") == 0 &&
                entry.path().extension() == ".ckpt") {
                checkpoints.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        return false; // Directory doesn't exist
    }
    
    if (checkpoints.empty()) {
        return false;
    }
    
    // Sort to find latest
    std::sort(checkpoints.begin(), checkpoints.end());
    std::string latest_checkpoint = checkpoints.back();
    
    // Extract episode number
    std::filesystem::path checkpoint_path(latest_checkpoint);
    std::string filename = checkpoint_path.stem().string();
    size_t pos = filename.find("checkpoint_episode_");
    if (pos != std::string::npos) {
        start_episode = std::stoi(filename.substr(pos + 19));
    }
    
    // Load agent (assuming we know the type - this could be improved)
    // For now, we'll need the agent to be pre-created
    if (agent) {
        agent->load_model(latest_checkpoint);
    }
    
    // Load metrics
    std::string metrics_path = exp_dir + "metrics_episode_" + std::to_string(start_episode) + ".csv";
    try {
        metrics.load_from_file(metrics_path);
    } catch (const std::exception&) {
        // Metrics file might not exist, continue anyway
    }
    
    std::cout << "Loaded checkpoint from episode " << start_episode << std::endl;
    return true;
}

void TrainingCheckpointManager::cleanup_old_checkpoints(const std::string& experiment_name) {
    std::string exp_dir = checkpoint_directory;
    if (!experiment_name.empty()) {
        exp_dir += experiment_name + "/";
    }
    
    std::vector<std::string> checkpoints;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(exp_dir)) {
            if (entry.is_regular_file() && 
                entry.path().filename().string().find("checkpoint_episode_") == 0) {
                checkpoints.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        return;
    }
    
    if ((int)checkpoints.size() <= max_checkpoints) {
        return;
    }
    
    // Sort by modification time
    std::sort(checkpoints.begin(), checkpoints.end(), [](const std::string& a, const std::string& b) {
        return std::filesystem::last_write_time(a) < std::filesystem::last_write_time(b);
    });
    
    // Remove oldest checkpoints
    int to_remove = checkpoints.size() - max_checkpoints;
    for (int i = 0; i < to_remove; i++) {
        try {
            std::filesystem::remove(checkpoints[i]);
            
            // Also remove associated files
            std::filesystem::path checkpoint_path(checkpoints[i]);
            std::string base_name = checkpoint_path.stem().string();
            std::string episode_str = base_name.substr(base_name.find("episode_"));
            
            std::string metrics_file = exp_dir + "metrics_" + episode_str + ".csv";
            std::string info_file = exp_dir + "checkpoint_info_" + episode_str + ".txt";
            
            std::filesystem::remove(metrics_file);
            std::filesystem::remove(info_file);
            
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error removing old checkpoint: " << e.what() << std::endl;
        }
    }
}

// ModelPersistenceUtils Implementation

namespace ModelPersistenceUtils {
    bool is_valid_model_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        std::string first_line;
        std::getline(file, first_line);
        file.close();
        
        // Check if it starts with a known agent type
        return (first_line == "RandomAgent" || 
                first_line == "QLearningAgent" || 
                first_line == "DQNAgent");
    }
    
    std::string sanitize_filename(const std::string& name) {
        std::string sanitized = name;
        
        // Replace invalid characters
        std::replace_if(sanitized.begin(), sanitized.end(), 
                       [](char c) { return c == '/' || c == '\\' || c == ':' || 
                                   c == '*' || c == '?' || c == '"' || 
                                   c == '<' || c == '>' || c == '|'; }, '_');
        
        return sanitized;
    }
    
    bool file_exists(const std::string& filepath) {
        return std::filesystem::exists(filepath);
    }
    
    void create_model_backup(const std::string& model_path, const std::string& backup_dir) {
        std::filesystem::create_directories(backup_dir);
        
        std::filesystem::path source(model_path);
        std::string backup_name = source.stem().string() + "_backup_" + 
                                 std::to_string(std::time(nullptr)) + source.extension().string();
        std::string backup_path = backup_dir + backup_name;
        
        try {
            std::filesystem::copy_file(model_path, backup_path);
            std::cout << "Backup created: " << backup_path << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating backup: " << e.what() << std::endl;
        }
    }
}
