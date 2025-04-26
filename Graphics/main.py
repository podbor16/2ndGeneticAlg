import numpy as np
import matplotlib.pyplot as plt

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
# Выбираем поколение (например, последнее)
front = data[-1]

plt.figure(figsize=(6,6))
plt.scatter(front[:,0], front[:,1], s=10, label='Найденный фронт')

# Теоретический фронт: f₂ = 1 - sqrt(f₁), 0 <= f₁ <= 1
f1 = np.linspace(0, 1, 200)
f2 = 1 - np.sqrt(f1)
plt.plot(f1, f2, 'r-', label='Опорный фронт')

plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Аппроксимация фронта Парето для задачи 3')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
