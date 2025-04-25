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
constexpr int N = 100;               // Размер популяции (число подзадач)
constexpr int T = 17;                // Число соседей
constexpr int MAX_GENERATIONS = 500; // Максимальное число поколений
constexpr int RUNS = 50;             // Количество независимых запусков

// -------------------------
// Параметры бинарного кодирования
// Для n = 3 переменных:
// x1 ∈ [0,1] (первый параметр),
// x2, x3 ∈ [-1,1] (оставшиеся параметры)
// -------------------------
constexpr double X0_MIN = 0.0, X0_MAX = 1.0;
constexpr double XK_MIN = -1.0, XK_MAX = 1.0;
constexpr int PRECISION = 16;
constexpr int IND_SIZE = 3 * PRECISION;
constexpr double EPS = 1e-6;

enum class CrossoverType { ONE_POINT, TWO_POINT, UNIFORM };
enum class MutationType  { WEAK, AVERAGE, STRONG };

constexpr CrossoverType CX_TYPE = CrossoverType::UNIFORM;
constexpr MutationType  MT_TYPE = MutationType::STRONG;

static mt19937 rng(random_device{}());
static uniform_real_distribution<double> dist01(0.0, 1.0);
static uniform_int_distribution<int> distBit(0, 1);

struct Individual {
    vector<int>    genes;
    vector<double> f; // Две цели
    Individual() : genes(IND_SIZE), f(2, 0.0) {}
};

// Функция декодирования бинарной строки в число из диапазона [lo, hi]
double decode(const vector<int>& g, int start, double lo, double hi) {
    long long val = 0;
    for (int i = 0; i < PRECISION; ++i)
        val = (val << 1) | g[start + i];
    return lo + (hi - lo) * val / ((1LL << PRECISION) - 1);
}

// -------------------------------------------------------------------------
// Функция оценки для второй задачи (см. формулы (5.10.6)–(5.10.8) учебного пособия)
// Пусть:
//   x1 = decode(genes[0...PRECISION-1]) ∈ [0, 1],
//   x2 = decode(genes[PRECISION...2*PRECISION-1]) ∈ [-1, 1],
//   x3 = decode(genes[2*PRECISION...3*PRECISION-1]) ∈ [-1, 1].
// В идеале должны выполняться равенства:
//    x2 = y2  и  x3 = y3,
// где
//    y2 = (0.3*x3*cos(2.47*x1+4.0) + 0.6*x1)* sin(6.7*x1+3.0)
//    y3 = (0.3*x3*cos(2.47*x1+8.0) + 0.6*x1)* cos(6.7*x1+3.0)
// Тогда:
//    f1 = x1 + |x2 - y2|
//    f2 = 1 - sqrt(x1) + |x3 - y3|
// При идеальном случае получаем: f1 = x1 и f2 = 1 - sqrt(x1),
// что и задаёт истинный фронт: f2 = 1 - sqrt(f1),  f1 ∈ [0,1].
// -------------------------------------------------------------------------
vector<double> evaluate(const Individual& ind) {
    double x1 = decode(ind.genes, 0, X0_MIN, X0_MAX);              // x1 ∈ [0,1]
    double x2 = decode(ind.genes, PRECISION, XK_MIN, XK_MAX);        // x2 ∈ [-1,1]
    double x3 = decode(ind.genes, 2 * PRECISION, XK_MIN, XK_MAX);    // x3 ∈ [-1,1]

    // Вычисление желаемого значения y2 для подмножества J2 (смещение = 4)
    double base_y2 = 0.3 * x3 * cos(2.47 * x1 + 4.0) + 0.6 * x1;
    double y2 = base_y2 * sin(6.7 * x1 + 3.0);

    // Вычисление желаемого значения y3 для подмножества J1 (смещение = 8)
    double base_y3 = 0.3 * x3 * cos(2.47 * x1 + 8.0) + 0.6 * x1;
    double y3 = base_y3 * cos(6.7 * x1 + 3.0);

    vector<double> f(2, 0.0);
    f[0] = x1 + fabs(x2 - y2);            // Целевая функция f1
    f[1] = 1.0 - sqrt(x1) + fabs(x3 - y3);  // Целевая функция f2
    return f;
}

// Генерация начальной популяции
vector<Individual> initPopulation() {
    vector<Individual> pop(N);
    for (auto& ind : pop) {
        for (int i = 0; i < IND_SIZE; ++i)
            ind.genes[i] = distBit(rng);
        ind.f = evaluate(ind);
    }
    return pop;
}

// Инициализация весовых векторов (равномерное разбиение для 2 целей)
vector<vector<double>> initWeights() {
    vector<vector<double>> W(N, vector<double>(2, 0.0));
    for (int i = 0; i < N; ++i) {
        double w1 = double(i) / (N - 1);
        W[i] = { w1, 1.0 - w1 };
    }
    return W;
}

// Инициализация соседей для каждого весового вектора (по Евклидову расстоянию)
vector<vector<int>> initNeighbors(const vector<vector<double>>& W) {
    vector<vector<int>> B(N);
    for (int i = 0; i < N; ++i) {
        vector<pair<double, int>> d;
        for (int j = 0; j < N; ++j) {
            double dx = W[i][0] - W[j][0];
            double dy = W[i][1] - W[j][1];
            d.push_back({ sqrt(dx * dx + dy * dy), j });
        }
        sort(d.begin(), d.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (int k = 0; k < T; ++k)
            B[i].push_back(d[k].second);
    }
    return B;
}

// Функция агрегирования на основе метода Чебышева
double chebyshev(const vector<double>& f, const vector<double>& w, const vector<double>& z) {
    double m = -numeric_limits<double>::infinity();
    for (int i = 0; i < 2; ++i)
        m = max(m, w[i] * fabs(f[i] - z[i]));
    return m;
}

// Функция для проверки доминирования (для минимизации)
// Возвращает true, если вектор u строго меньше v хотя бы по одному критерию и не хуже по остальных.
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

// ---------------------------
// Операторы кроссовера
// ---------------------------
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
    if (p1 > p2) swap(p1, p2);
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

// ---------------------------
// Оператор мутации
// ---------------------------
void mutate(Individual& ind, MutationType mt) {
    double pm = 0.0;
    if (mt == MutationType::WEAK)    pm = 1.0 / (3 * IND_SIZE);
    if (mt == MutationType::AVERAGE) pm = 1.0 / IND_SIZE;
    if (mt == MutationType::STRONG)  pm = min(3.0 / double(IND_SIZE), 1.0);
    for (int i = 0; i < IND_SIZE; ++i)
        if (dist01(rng) < pm)
            ind.genes[i] = 1 - ind.genes[i];
}

// -------------------------------------------------------------------------
// Генерация опорного (истинного) фронта Парето для IGD для второй задачи.
// В идеале, когда x2=y2 и x3=y3, получаем f1=x1 и f2=1-sqrt(x1),
// т.е. истинный фронт: f2 = 1 - sqrt(f1), для f1 ∈ [0,1].
// -------------------------------------------------------------------------
vector<vector<double>> buildReference() {
    vector<vector<double>> P;
    P.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        double x = i / 999.0;
        vector<double> point(2, 0.0);
        point[0] = x;
        point[1] = 1.0 - sqrt(x);
        P.push_back(point);
    }
    return P;
}

// Вычисление IGD (Inverted Generational Distance)
double computeIGD(const vector<vector<double>>& P, const vector<vector<double>>& A) {
    double sum = 0.0;
    for (const auto& v : P) {
        double dmin = numeric_limits<double>::infinity();
        for (const auto& u : A) {
            double d = hypot(v[0] - u[0], v[1] - u[1]);
            dmin = min(dmin, d);
        }
        sum += dmin;
    }
    return sum / P.size();
}

// ============================
// Главная функция (тестирование второй задачи)
// ============================
int main() {
    auto W = initWeights();
    auto B = initNeighbors(W);
    auto ref = buildReference();
    double bestIGD = numeric_limits<double>::infinity();
    vector<vector<double>> best_EP; // архив недоминируемых решений лучшего запуска

    // Проведение независимых запусков
    for (int run = 0; run < RUNS; ++run) {
        // Инициализация популяции и идеального вектора z
        auto pop = initPopulation();
        vector<double> z(2, numeric_limits<double>::infinity());
        for (const auto& ind : pop) {
            for (int j = 0; j < 2; ++j)
                z[j] = min(z[j], ind.f[j]);
        }
        // Внешний архив для хранения недоминируемых решений данного запуска
        vector<vector<double>> EP;

        // Основной цикл поколений
        for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
            for (int i = 0; i < N; ++i) {
                // Выбор двух родителей из окрестности i
                int idx1 = B[i][int(dist01(rng) * T)];
                int idx2 = B[i][int(dist01(rng) * T)];
                Individual child;
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
                // Обновление идеальной точки z
                for (int j = 0; j < 2; ++j)
                    z[j] = min(z[j], child.f[j]);
                // Обновление соседей: для каждого nb из окрестности i вычисляем агрегированное значение с весом W[nb]
                for (int nb : B[i]) {
                    double f_child = chebyshev(child.f, W[nb], z);
                    double f_nb = chebyshev(pop[nb].f, W[nb], z);
                    if (f_child < f_nb) {
                        pop[nb] = child;
                    }
                }
            }
            // Извлечение текущего недоминируемого фронта из популяции
            vector<vector<double>> pf;
            for (int i = 0; i < N; ++i) {
                bool domFlag = false;
                for (int j = 0; j < N; ++j) {
                    if (i != j && dominates(pop[j].f, pop[i].f)) {
                        domFlag = true;
                        break;
                    }
                }
                if (!domFlag)
                    pf.push_back(pop[i].f);
            }
            // Обновление внешнего архива EP недоминируемыми решениями
            for (const auto& p : pf) {
                EP.erase(remove_if(EP.begin(), EP.end(),
                                   [&](const vector<double>& e) { return dominates(p, e); }),
                         EP.end());
                bool dominatedByEP = false;
                for (const auto& e : EP) {
                    if (dominates(e, p)) { dominatedByEP = true; break; }
                }
                if (!dominatedByEP)
                    EP.push_back(p);
            }
            // (Опционально: можно выводить динамику IGD по поколениям)
            // double igd = computeIGD(ref, EP);
            // cout << "Generation " << gen+1 << " IGD = " << igd << "\n";
        }
        double runIGD = computeIGD(ref, EP);
        cout << runIGD << "\n";
        if (runIGD < bestIGD) {
            bestIGD = runIGD;
            best_EP = EP;
        }
    }
    cout << "Best IGD over all runs: " << bestIGD << endl;

    // Сохранение лучшего найденного фронта Парето в файл
    ofstream outFile("pareto_final.txt");
    if (outFile.is_open()) {
        for (const auto& p : best_EP)
            outFile << p[0] << " " << p[1] << "\n";
        outFile.close();
        cout << "Final Pareto front saved to pareto_final.txt" << endl;
    } else {
        cerr << "Error opening output file!" << endl;
    }
    return 0;
}
