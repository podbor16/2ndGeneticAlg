import numpy as np
import matplotlib.pyplot as plt

# Число переменных, используемое в задаче – NP = 10
NP = 10

def read_data(file_path):
    gens = []
    with open(file_path) as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                if data:
                    gens.append(np.array(data))
                    data = []
            else:
                x, y = map(float, line.split())
                data.append([x, y])
        if data:
            gens.append(np.array(data))
    return gens

data = read_data('pareto_final.txt')
front = data[-1]

plt.figure(figsize=(6,6))
plt.scatter(front[:,0], front[:,1], s=10, label='Найденный фронт')

# Генерируем истинный фронт (опорный) согласно формуле:
# f1 = i/(2NP), f2 = 1 - f1, для i = 0, 1, …, 2NP
num_points = int(2*NP + 1)
f1 = np.array([i/(2*NP) for i in range(num_points)])
f2 = 1 - f1
plt.scatter(f1, f2, s=20, color='red', label='Опорный фронт')  # Используем scatter вместо plot

plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Аппроксимация фронта Парето для задачи 5')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
