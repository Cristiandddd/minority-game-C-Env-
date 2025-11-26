/***************************************************************************
                          model_persistence.h  -  description
                             -------------------
    begin                : December 2024
    email                : 
 ***************************************************************************/

#ifndef _MODEL_PERSISTENCE_H_
#define _MODEL_PERSISTENCE_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <chrono>
#include "rl_agents.h"
#include "training_framework.h"

// Model metadata for enhanced persistence
struct ModelMetadata {
    std::string agent_type;
    std::string creation_timestamp;
    std::string version;
    int observation_space_size;
    int action_space_size;
    std::map<std::string, double> hyperparameters;
    std::map<std::string, std::string> training_info;
    
    // Performance metrics at save time
    double final_average_reward;
    double final_win_rate;
    int training_episodes;
    double training_time_seconds;
    
    ModelMetadata() : observation_space_size(0), action_space_size(0),
                     final_average_reward(0.0), final_win_rate(0.0),
                     training_episodes(0), training_time_seconds(0.0) {
        version = "1.0";
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        creation_timestamp = std::to_string(time_t);
    }
    
    void save_to_stream(std::ofstream& file) const;
    void load_from_stream(std::ifstream& file);
    void print_info() const;
};

// Enhanced model persistence manager
class ModelPersistenceManager {
private:
    std::string base_path;
    std::string models_directory;
    std::string metadata_directory;
    
    void ensure_directories_exist();
    std::string generate_model_filename(const std::string& agent_type, 
                                       const std::string& suffix = "") const;
    std::string generate_metadata_filename(const std::string& model_filename) const;
    
public:
    ModelPersistenceManager(const std::string& base_path = "saved_models/");
    
    // Enhanced save/load with metadata
    void save_model_with_metadata(RLAgent* agent, const ModelMetadata& metadata,
                                 const std::string& custom_name = "");
    std::unique_ptr<RLAgent> load_model_with_metadata(const std::string& model_path,
                                                     ModelMetadata& metadata);
    
    // Batch operations
    void save_multiple_models(const std::vector<std::unique_ptr<RLAgent>>& agents,
                             const std::vector<ModelMetadata>& metadata_list,
                             const std::string& batch_name = "");
    std::vector<std::unique_ptr<RLAgent>> load_multiple_models(
        const std::vector<std::string>& model_paths,
        std::vector<ModelMetadata>& metadata_list);
    
    // Model management utilities
    std::vector<std::string> list_available_models() const;
    std::vector<ModelMetadata> get_models_metadata() const;
    bool delete_model(const std::string& model_path);
    void cleanup_old_models(int keep_latest_n = 10);
    
    // Model comparison and selection
    std::string find_best_model(const std::string& agent_type, 
                               const std::string& metric = "final_average_reward") const;
    std::vector<std::string> find_models_by_criteria(
        const std::map<std::string, std::string>& criteria) const;
    
    // Export/Import functionality
    void export_model_package(const std::string& model_path, 
                             const std::string& export_path) const;
    void import_model_package(const std::string& package_path,
                             const std::string& import_path = "");
    
    // Getters
    std::string get_models_directory() const { return models_directory; }
    std::string get_metadata_directory() const { return metadata_directory; }
};

// Model versioning and compatibility
class ModelVersionManager {
private:
    static const std::string CURRENT_VERSION;
    
public:
    static bool is_compatible(const std::string& model_version);
    static std::string get_current_version() { return CURRENT_VERSION; }
    static void upgrade_model_if_needed(const std::string& model_path);
};

// Checkpoint system for long training runs
class TrainingCheckpointManager {
private:
    std::string checkpoint_directory;
    int checkpoint_frequency;
    int max_checkpoints;
    
public:
    TrainingCheckpointManager(const std::string& checkpoint_dir = "checkpoints/",
                             int frequency = 100, int max_checkpoints = 5);
    
    // Checkpoint operations
    void save_checkpoint(RLAgent* agent, const TrainingMetrics& metrics,
                        int episode, const std::string& experiment_name = "");
    bool load_latest_checkpoint(std::unique_ptr<RLAgent>& agent,
                               TrainingMetrics& metrics, int& start_episode,
                               const std::string& experiment_name = "");
    
    // Checkpoint management
    std::vector<std::string> list_checkpoints(const std::string& experiment_name = "") const;
    void cleanup_old_checkpoints(const std::string& experiment_name = "");
    bool resume_training_possible(const std::string& experiment_name = "") const;
    
    // Getters/Setters
    void set_checkpoint_frequency(int frequency) { checkpoint_frequency = frequency; }
    void set_max_checkpoints(int max_checkpoints) { this->max_checkpoints = max_checkpoints; }
};

// Model performance tracking
struct ModelPerformanceRecord {
    std::string model_path;
    std::string agent_type;
    std::string timestamp;
    double average_reward;
    double win_rate;
    double std_reward;
    double std_win_rate;
    int evaluation_episodes;
    std::map<std::string, double> additional_metrics;
    
    ModelPerformanceRecord() : average_reward(0.0), win_rate(0.0),
                              std_reward(0.0), std_win_rate(0.0),
                              evaluation_episodes(0) {}
};

class ModelPerformanceTracker {
private:
    std::string performance_log_path;
    std::vector<ModelPerformanceRecord> performance_history;
    
public:
    ModelPerformanceTracker(const std::string& log_path = "performance_log.csv");
    
    // Performance tracking
    void record_performance(const ModelPerformanceRecord& record);
    void evaluate_and_record(RLAgent* agent, const std::string& model_path,
                           MinorityGameEnv& env, int num_episodes = 100);
    
    // Analysis
    std::vector<ModelPerformanceRecord> get_top_performers(int n = 10,
        const std::string& metric = "average_reward") const;
    ModelPerformanceRecord get_best_performance(const std::string& agent_type = "",
        const std::string& metric = "average_reward") const;
    
    // Reporting
    void generate_performance_report(const std::string& output_path) const;
    void plot_performance_trends(const std::string& output_path) const;
    
    // Data management
    void save_performance_log() const;
    void load_performance_log();
    void clear_performance_history() { performance_history.clear(); }
};

// Utility functions for model persistence
namespace ModelPersistenceUtils {
    // File format validation
    bool is_valid_model_file(const std::string& filepath);
    std::string get_model_format_version(const std::string& filepath);
    
    // Path utilities
    std::string sanitize_filename(const std::string& name);
    std::string get_file_extension(const std::string& filepath);
    bool file_exists(const std::string& filepath);
    
    // Compression utilities (for large models)
    void compress_model_file(const std::string& input_path, const std::string& output_path);
    void decompress_model_file(const std::string& input_path, const std::string& output_path);
    
    // Model integrity
    bool verify_model_integrity(const std::string& filepath);
    std::string calculate_model_checksum(const std::string& filepath);
    
    // Backup utilities
    void create_model_backup(const std::string& model_path, const std::string& backup_dir = "backups/");
    void restore_model_from_backup(const std::string& backup_path, const std::string& restore_path);
}

#endif // _MODEL_PERSISTENCE_H_
