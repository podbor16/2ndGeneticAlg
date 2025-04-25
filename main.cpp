#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <numeric>

using namespace std;

// Параметры алгоритма
constexpr int N = 100;               // Размер популяции
constexpr int T = 17;                // Число соседей
constexpr int MAX_GENERATIONS = 500; // Максимальное число поколений
constexpr int RUNS = 50;             // Количество независимых запусков

// Параметры бинарного кодирования (3 переменные: первая в [0,1], остальные в [-1,1])
constexpr double X0_MIN = 0.0, X0_MAX = 1.0;
constexpr double XK_MIN = -1.0, XK_MAX = 1.0;
constexpr int PRECISION = 16;
constexpr int IND_SIZE = 3 * PRECISION; // 3 переменные, каждая кодируется PRECISION битами
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
    vector<double> f;
    Individual() : genes(IND_SIZE), f(2, 0.0) {}
};

// Функция декодирования бинарной строки в число
double decode(const vector<int>& g, int start, double lo, double hi) {
    long long val = 0;
    for (int i = 0; i < PRECISION; ++i)
        val = (val << 1) | g[start + i];
    return lo + (hi - lo) * val / ((1LL << PRECISION) - 1);
}

// Функция оценки. Реализована для задачи 1 согласно пособию.
// Пусть:
//  x0 = x[0] ∈ [0,1]  – первый параметр;
//  x1 = x[1] ∈ [-1,1] – должен стремиться к sin(6π*x0+2π/3);
//  x2 = x[2] ∈ [-1,1] – должен стремиться к sin(6π*x0+π).
// Тогда целевые функции:
//  f1 = x0 + |x1 - sin(6πx0+2π/3)|
//  f2 = 1 - sqrt(x0) + |x2 - sin(6πx0+π)|
vector<double> evaluate(const Individual& ind) {
    double x0 = decode(ind.genes, 0, X0_MIN, X0_MAX);
    double x1 = decode(ind.genes, PRECISION, XK_MIN, XK_MAX);
    double x2 = decode(ind.genes, 2 * PRECISION, XK_MIN, XK_MAX);
    vector<double> f(2, 0.0);
    f[0] = x0 + fabs(x1 - sin(6 * M_PI * x0 + 2 * M_PI / 3));
    f[1] = 1.0 - sqrt(x0) + fabs(x2 - sin(6 * M_PI * x0 + M_PI));
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

// Инициализация весовых векторов
vector<vector<double>> initWeights() {
    vector<vector<double>> W(N, vector<double>(2, 0.0));
    for (int i = 0; i < N; ++i) {
        double w1 = double(i) / (N - 1);
        W[i] = {w1, 1.0 - w1};
    }
    return W;
}

// Инициализация соседей для каждого весового вектора по Евклидову расстоянию
vector<vector<int>> initNeighbors(const vector<vector<double>>& W) {
    vector<vector<int>> B(N);
    for (int i = 0; i < N; ++i) {
        vector<pair<double, int>> d;
        for (int j = 0; j < N; ++j) {
            double dx = W[i][0] - W[j][0];
            double dy = W[i][1] - W[j][1];
            d.push_back({sqrt(dx * dx + dy * dy), j});
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

// Функция для проверки доминирования (для минимизации). Возвращает true, если вектор u строго лучше по крайней мере по одному критерию и не хуже по оставшимся.
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

// Операторы кроссовера
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

// Оператор мутации
void mutate(Individual& ind, MutationType mt) {
    double pm = 0.0;
    if (mt == MutationType::WEAK)    pm = 1.0 / (3 * IND_SIZE);
    if (mt == MutationType::AVERAGE) pm = 1.0 / IND_SIZE;
    if (mt == MutationType::STRONG)  pm = min(3.0 / double(IND_SIZE), 1.0);
    for (int i = 0; i < IND_SIZE; ++i)
        if (dist01(rng) < pm)
            ind.genes[i] = 1 - ind.genes[i];
}

// Генерация опорного (истинного) фронта Парето для IGD (для задачи 1 — 1000 точек).
// В условном идеале для задачи 1 оптимальные значения достигаются, если x2 = sin(6πx1+2π/3) и x3 = sin(6πx1+π),
// что даёт f1 = x1 и f2 = 1 - sqrt(x1).
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

int main() {
    auto W = initWeights();
    auto B = initNeighbors(W);
    auto ref = buildReference();
    double bestIGD = numeric_limits<double>::infinity();
    vector<vector<double>> best_EP; // архив недоминируемых решений лучшего запуска

    // Независимые запуски
    for (int run = 0; run < RUNS; ++run) {
        //cout << "Run " << run + 1 << " / " << RUNS << endl;
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
                // Обновление идеального вектора z
                for (int j = 0; j < 2; ++j)
                    z[j] = min(z[j], child.f[j]);

                // Обновление соседей.
                // Для каждого соседа nb из окрестности i вычисляем агрегированное значение с весовым вектором W[nb].
                for (int nb : B[i]) {
                    double f_child = chebyshev(child.f, W[nb], z);
                    double f_nb = chebyshev(pop[nb].f, W[nb], z);
                    if (f_child < f_nb) {
                        pop[nb] = child;
                    }
                }
            }
            // Извлечение текущего фронта (недоминированных решений) из популяции
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
            // Обновление внешнего архива EP недоминированными решениями
            for (const auto& p : pf) {
                // Удаляем из EP решения, доминируемые p
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
        //cout << "Run " << run + 1 << " final IGD: " << runIGD << "\n";
        cout << runIGD << "\n";
        if (runIGD < bestIGD) {
            bestIGD = runIGD;
            best_EP = EP;
        }
    }
    cout << "Best IGD over all runs: " << bestIGD << endl;

    // Сохранение результатов (лучший найденный фронт Парето) в файл
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
