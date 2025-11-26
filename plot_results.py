import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Cargar CSV (ajusta el nombre al tuyo)
df = pd.read_csv("sweep_results.csv")

# Calcular alpha = 2^memory_size / num_players
df['alpha'] = (2 ** df['memory_size']) / df['num_players']

# Ordenar por alpha para que los plots sean más claros
df = df.sort_values(by="alpha")

# Procesar por cada memory_size distinto en el CSV
for memory in df['memory_size'].unique():
    sub_df = df[df['memory_size'] == memory]

    # Crear carpeta destino
    folder = f"sweep_results/m{memory}"
    os.makedirs(folder, exist_ok=True)

    # --- Gráfico 1: Average Reward vs Alpha ---
    plt.errorbar(sub_df['alpha'], sub_df['average_reward'], yerr=sub_df['std_reward'],
                 fmt='o-', capsize=5, label="Average Reward")
    plt.xlabel("Alpha = 2^M / N")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward vs Alpha (M={memory})")
    plt.xscale('log')  # Escala logarítmica en x
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/reward_vs_alpha.png")
    plt.close()

    # --- Gráfico 2: Win Rate vs Alpha (con Non-RL Win Rate) ---
    plt.figure(figsize=(10, 6))
    
    # Win Rate del agente RL
    plt.errorbar(sub_df['alpha'], sub_df['win_rate'], yerr=sub_df['std_win_rate'],
                 fmt='s-', capsize=5, color="blue", label="RL Win Rate", linewidth=2, markersize=6)
    
    # Non-RL Win Rate
    plt.errorbar(sub_df['alpha'], sub_df['non_rl_avg_win_rate'], yerr=sub_df['non_rl_std_win_rate'],
                 fmt='^-', capsize=5, color="red", label="Non-RL Win Rate", linewidth=2, markersize=6)
    
    plt.xlabel("Alpha = 2^M / N")
    plt.ylabel("Win Rate")
    plt.title(f"Win Rate vs Alpha (M={memory})")
    plt.xscale('log')  # Escala logarítmica en x
    plt.axvline(x=0.34, color='red', linestyle='--', linewidth=2, alpha=0.7, label='α=0.34')
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/winrate_vs_alpha.png", dpi=300)
    plt.close()

    print(f"Gráficos guardados en: {folder}")

# --- Gráfico adicional: Comparación de todos los memory_size juntos ---
plt.figure(figsize=(12, 8))

for memory in df['memory_size'].unique():
    sub_df = df[df['memory_size'] == memory].sort_values(by="alpha")
    plt.plot(sub_df['alpha'], sub_df['win_rate'], 'o-', label=f"RL M={memory}", linewidth=2)
    plt.plot(sub_df['alpha'], sub_df['non_rl_avg_win_rate'], '--', alpha=0.7, label=f"Non-RL M={memory}")

plt.xlabel("Alpha = 2^M / N")
plt.ylabel("Win Rate")
plt.title("Win Rate vs Alpha - Comparación de todos los Memory Size")
plt.xscale('log')
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("sweep_results/all_memory_comparison.png", dpi=300)
plt.close()

print("Gráfico de comparación guardado en: sweep_results/all_memory_comparison.png")