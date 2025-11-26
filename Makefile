# Makefile for Minority Game RL Training System

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -I.
LIBS = -lstdc++fs

# Source files
SOURCES = minority.cpp agent.cpp rnd.cpp minority_game_env.cpp rl_agents.cpp training_framework.cpp train.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = train

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Create necessary directories
setup:
	mkdir -p models
	mkdir -p metrics
	mkdir -p logs

# Install dependencies (if needed)
install-deps:
	# Add any dependency installation commands here
	@echo "No external dependencies required"

# Run tests
test: $(TARGET)
	./$(TARGET) --agent random --episodes 10 --verbose
	./$(TARGET) --agent qlearning --episodes 50 --verbose
	./$(TARGET) --compare --episodes 100

# Run example training sessions
examples: $(TARGET) setup
	@echo "Running example training sessions..."
	./$(TARGET) --agent qlearning --episodes 200 --players 51 --memory 2
	./$(TARGET) --agent dqn --episodes 200 --players 51 --memory 2
	./$(TARGET) --multiagent 2 --episodes 100 --players 51
	./$(TARGET) --compare --episodes 100

# Debug build
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Release build
release: CXXFLAGS += -DNDEBUG -O3
release: clean $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the training executable (default)"
	@echo "  clean     - Remove build files"
	@echo "  setup     - Create necessary directories"
	@echo "  test      - Run basic tests"
	@echo "  examples  - Run example training sessions"
	@echo "  debug     - Build with debug flags"
	@echo "  release   - Build optimized release version"
	@echo "  help      - Show this help message"

.PHONY: all clean setup install-deps test examples debug release help
