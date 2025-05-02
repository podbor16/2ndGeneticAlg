#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <numeric>

using namespace std;

// -------------------------
// Параметры алгоритма MOEA/D
// -------------------------
constexpr int POP_SIZE = 100;               // Размер популяции (число подзадач)
constexpr int T = 17;                       // Число соседей
constexpr int MAX_GENERATIONS = 500;        // Максимальное число поколений
constexpr int RUNS = 50;                    // Количество независимых запусков

// -------------------------
// Параметры задачи 5
// -------------------------
constexpr int PROBLEM_N = 10;        // Общее число переменных в задаче (n)
constexpr double delta = 0.1;        // Смещение фазы
constexpr double NP = 10.0;          // Параметр для дискретизации истинного фронта (N_P)
constexpr double alpha = 1.0 / (2.0 * NP);  // Коэффициент штрафа (0.05)

// -------------------------
// Параметры бинарного кодирования
// -------------------------
constexpr double X0_MIN = 0.0, X0_MAX = 1.0;    // x1 ∈ [0,1]
constexpr double XK_MIN = -1.0, XK_MAX = 1.0;    // x_j, j>=2, ∈ [-1,1]
constexpr int PRECISION = 16;                     // Число бит для каждой переменной
constexpr int IND_SIZE = PROBLEM_N * PRECISION;   // Общий размер индивида (n*PRECISION)
constexpr double EPS = 1e-6;

enum class CrossoverType { ONE_POINT, TWO_POINT, UNIFORM };
enum class MutationType  { WEAK, AVERAGE, STRONG };

constexpr CrossoverType CX_TYPE = CrossoverType::UNIFORM;
constexpr MutationType MT_TYPE = MutationType::STRONG;

static mt19937 rng(random_device{}());
static uniform_real_distribution<double> dist01(0.0, 1.0);
static uniform_int_distribution<int> distBit(0, 1);

// Структура индивида
struct Individual {
    vector<int> genes;
    vector<double> f;  // Две цели: f1 и f2
    Individual() : genes(IND_SIZE), f(2, 0.0) {}
};

// Функция декодирования: переводит отрезок бинарной строки, начиная с index, в число из интервала [lo, hi]
double decode(const vector<int>& g, int index, double lo, double hi) {
    long long val = 0;
    for (int i = 0; i < PRECISION; ++i)
        val = (val << 1) | g[index + i];
    return lo + (hi - lo) * val / ((1LL << PRECISION) - 1);
}

// Функция h(y) = 2*y^2 - cos(4*pi*y) + 1
double h(double y) {
    return 2 * y * y - cos(4 * M_PI * y) + 1;
}

// Функция оценки для задачи 5
// Декодируются: x1 ∈ [0,1] (из генов [0, PRECISION-1]) и x2...x_n из [-1,1]
vector<double> evaluate(const Individual& ind) {
    double x1 = decode(ind.genes, 0, X0_MIN, X0_MAX);
    vector<double> x(PROBLEM_N);
    x[0] = x1;
    for (int j = 1; j < PROBLEM_N; ++j) {
        x[j] = decode(ind.genes, j * PRECISION, XK_MIN, XK_MAX);
    }
    // Разобьём переменные от j=1 до n-1 на два подмножества:
    // Пусть J1: переменные с четными индексами (j % 2 == 0)
    //      J2: переменные с нечетными индексами (j % 2 == 1)
    double sumJ1 = 0.0, sumJ2 = 0.0, sumH = 0.0;
    for (int j = 1; j < PROBLEM_N; ++j) {
        if (j % 2 == 0) {
            sumJ1 += fabs(sin(2 * NP * M_PI * x[j]));
        } else {
            sumJ2 += fabs(sin(2 * NP * M_PI * x[j]));
            sumH += h(x[j]);
        }
    }
    vector<double> f(2, 0.0);
    f[0] = x1 + (1.0 / (2 * NP)) * sumJ1;
    f[1] = 1.0 - x1 + (1.0 / (2 * NP)) * sumJ2 + (1.0 / NP) * sumH;
    return f;
}

// Генерация начальной популяции
vector<Individual> initPopulation() {
    vector<Individual> pop(POP_SIZE);
    for (auto &ind : pop) {
        for (int i = 0; i < IND_SIZE; ++i)
            ind.genes[i] = distBit(rng);
        ind.f = evaluate(ind);
    }
    return pop;
}

// Генерация равномерно распределённых весовых векторов для 2 целей
vector<vector<double>> initWeights() {
    vector<vector<double>> W(POP_SIZE, vector<double>(2, 0.0));
    for (int i = 0; i < POP_SIZE; ++i) {
        double w1 = double(i) / (POP_SIZE - 1);
        W[i] = {w1, 1.0 - w1};
    }
    return W;
}

// Инициализация соседей для каждого весового вектора (по евклидову расстоянию)
vector<vector<int>> initNeighbors(const vector<vector<double>> &W) {
    vector<vector<int>> B(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        vector<pair<double, int>> d;
        for (int j = 0; j < POP_SIZE; ++j) {
            double dx = W[i][0] - W[j][0];
            double dy = W[i][1] - W[j][1];
            d.push_back({sqrt(dx * dx + dy * dy), j});
        }
        sort(d.begin(), d.end(), [](auto &a, auto &b) { return a.first < b.first; });
        for (int k = 0; k < T; ++k)
            B[i].push_back(d[k].second);
    }
    return B;
}

// Функция агрегирования (метод Чебышева)
double chebyshev(const vector<double>& f, const vector<double>& w, const vector<double>& z) {
    double m = -numeric_limits<double>::infinity();
    for (int i = 0; i < 2; ++i)
        m = max(m, w[i] * fabs(f[i] - z[i]));
    return m;
}

// Функция доминирования: возвращает true, если вектор u доминирует v
bool dominates(const vector<double>& u, const vector<double>& v) {
    bool strictlyBetter = false;
    for (int i = 0; i < 2; ++i) {
        if (u[i] > v[i])
            return false;
        if (u[i] < v[i])
            strictlyBetter = true;
    }
    return strictlyBetter;
}

// Генетические операторы
Individual onePoint(const Individual& a, const Individual& b) {
    Individual c;
    int pt = 1 + int(dist01(rng) * (IND_SIZE - 2));
    for (int i = 0; i < pt; ++i)
        c.genes[i] = a.genes[i];
    for (int i = pt; i < IND_SIZE; ++i)
        c.genes[i] = b.genes[i];
    return c;
}

Individual twoPoint(const Individual& a, const Individual& b) {
    Individual c;
    int p1 = 1 + int(dist01(rng) * (IND_SIZE - 2));
    int p2 = 1 + int(dist01(rng) * (IND_SIZE - 2));
    if(p1 > p2)
        swap(p1, p2);
    for (int i = 0; i < p1; ++i)
        c.genes[i] = a.genes[i];
    for (int i = p1; i < p2; ++i)
        c.genes[i] = b.genes[i];
    for (int i = p2; i < IND_SIZE; ++i)
        c.genes[i] = a.genes[i];
    return c;
}

Individual uniCross(const Individual& a, const Individual& b) {
    Individual c;
    for (int i = 0; i < IND_SIZE; ++i)
        c.genes[i] = (distBit(rng) ? a.genes[i] : b.genes[i]);
    return c;
}

// Оператор мутации
void mutate(Individual &ind, MutationType mt) {
    double pm = 0.0;
    if(mt == MutationType::WEAK)
        pm = 1.0 / (3 * IND_SIZE);
    else if(mt == MutationType::AVERAGE)
        pm = 1.0 / IND_SIZE;
    else if(mt == MutationType::STRONG)
        pm = min(3.0 / double(IND_SIZE), 1.0);
    for (int i = 0; i < IND_SIZE; ++i) {
        if(dist01(rng) < pm)
            ind.genes[i] = 1 - ind.genes[i];
    }
}

// Функция computeIGD – вычисляет Inverted Generational Distance
double computeIGD(const vector<vector<double>>& P, const vector<vector<double>>& A) {
    double sum = 0.0;
    for (const auto &v : P) {
        double dmin = numeric_limits<double>::infinity();
        for (const auto &u : A) {
            double d = hypot(v[0] - u[0], v[1] - u[1]);
            dmin = min(dmin, d);
        }
        sum += dmin;
    }
    return sum / P.size();
}

// Генерация истинного (референсного) фронта Парето для задачи 5.
// Согласно учебному пособию, истинный фронт определяется дискретно:
// f1 = i/(2NP), f2 = 1 - f1, i = 0, 1, ..., 2NP
vector<vector<double>> buildReference() {
    vector<vector<double>> P;
    int numPoints = 2 * NP + 1; // 2N_P + 1 точка
    for (int i = 0; i < numPoints; ++i) {
        double f1 = double(i) / (2 * NP);
        vector<double> point(2, 0.0);
        point[0] = f1;
        point[1] = 1.0 - f1;
        P.push_back(point);
    }
    return P;
}

// ============================
// Основной цикл MOEA/D для задачи 5
// ============================
int main() {
    auto W = initWeights();
    auto B = initNeighbors(W);
    auto ref = buildReference();
    double bestIGD = numeric_limits<double>::infinity();
    vector<vector<double>> best_EP; // Лучший внешний архив (EP)

    for (int run = 0; run < RUNS; ++run) {
        auto pop = initPopulation();

        // Инициализируем идеальную точку z как минимальное значение по каждой цели в начальной популяции
        vector<double> z(2, numeric_limits<double>::infinity());
        for (const auto &ind : pop) {
            for (int j = 0; j < 2; ++j) {
                z[j] = min(z[j], ind.f[j]);
            }
        }
        vector<vector<double>> EP; // Внешний архив для данного запуска

        for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
            for (int i = 0; i < POP_SIZE; ++i) {
                // Выбор двух случайных соседей из набора B[i]
                int idx1 = B[i][int(dist01(rng) * T)];
                int idx2 = B[i][int(dist01(rng) * T)];
                Individual child;
                // Применяем оператор кроссовера
                switch (CX_TYPE) {
                    case CrossoverType::ONE_POINT:
                        child = onePoint(pop[idx1], pop[idx2]);
                        break;
                    case CrossoverType::TWO_POINT:
                        child = twoPoint(pop[idx1], pop[idx2]);
                        break;
                    default:
                        child = uniCross(pop[idx1], pop[idx2]);
                        break;
                }
                mutate(child, MT_TYPE);
                child.f = evaluate(child);
                // Обновляем идеальную точку z
                for (int j = 0; j < 2; ++j)
                    z[j] = min(z[j], child.f[j]);
                // Обновляем решения в окрестности
                for (int nb : B[i]) {
                    double f_child = chebyshev(child.f, W[nb], z);
                    double f_nb = chebyshev(pop[nb].f, W[nb], z);
                    if (f_child < f_nb)
                        pop[nb] = child;
                }
            }
            // Извлекаем недоминированный фронт (ПФ) из текущей популяции
            vector<vector<double>> pf;
            for (int i = 0; i < POP_SIZE; ++i) {
                bool isDominated = false;
                for (int j = 0; j < POP_SIZE; ++j) {
                    if (i != j && dominates(pop[j].f, pop[i].f)) {
                        isDominated = true;
                        break;
                    }
                }
                if (!isDominated)
                    pf.push_back(pop[i].f);
            }
            // Обновляем внешний архив EP
            for (const auto &p : pf) {
                EP.erase(remove_if(EP.begin(), EP.end(),
                                   [&](const vector<double>& e) { return dominates(p, e); }),
                         EP.end());
                bool dominatedByEP = false;
                for (const auto &e : EP) {
                    if (dominates(e, p)) { dominatedByEP = true; break; }
                }
                if (!dominatedByEP)
                    EP.push_back(p);
            }
        } // Конец поколений

        double runIGD = computeIGD(ref, EP);
        //cout << "Run " << run + 1 << ": IGD = " << runIGD << "\n";
        cout << runIGD << "\n";
        if (runIGD < bestIGD) {
            bestIGD = runIGD;
            best_EP = EP;
        }
    } // Конец независимых запусков

    cout << "Best IGD over all runs: " << bestIGD << "\n";

    // Сохраняем лучший найденный фронт Парето
    ofstream outFile("pareto_final.txt");
    if (outFile.is_open()) {
        for (const auto &p : best_EP) {
            outFile << p[0] << " " << p[1] << "\n";
        }
        outFile.close();
        cout << "Final Pareto front saved to pareto_final.txt\n";
    } else {
        cerr << "Error opening output file!\n";
    }

    return 0;
}
