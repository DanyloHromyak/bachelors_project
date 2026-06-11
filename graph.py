import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('pso_log.csv')
plt.figure(figsize=(8, 4))
plt.plot(data['iteration'], data['f1'], marker='o', linestyle='-', color='black')
plt.xlabel('PSO Iterations')
plt.ylabel('Best Macro-F1 Score')
plt.grid(True)
plt.savefig('pso_convergence.pdf')