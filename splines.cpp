#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <iomanip>
#include <string_view>
#include <utility>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <float.h>


using Point = std::pair<double, double>;
using Array2D = std::pair<std::vector<double>, std::vector<double>>;
// Bézier curve degree
const size_t DEGREE = 3;

std::ostream& operator<<(std::ostream& os, const Point& p)
{
    os << "(" << std::setprecision(15) << p.first << ", "
       << std::setprecision(15) << p.second << ")";
    return os;
}

Point operator*(const Point& lhs, double t)
{
    return Point{ lhs.first * t, lhs.second * t };
}

Point operator*(double t, const Point& lhs)
{
    return lhs * t;
}

Point operator/(const Point& lhs, double t)
{
    return Point{ lhs.first / t, lhs.second / t };
}

Point operator+(const Point& lhs, const Point& rhs)
{
    return Point{ lhs.first + rhs.first, lhs.second + rhs.second };
}

Point operator-(const Point& lhs, const Point& rhs)
{
    return Point{ lhs.first - rhs.first, lhs.second - rhs.second };
}

// https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
// Evaluate curve on parameter value `t`
Point de_casteljau(std::vector<Point>& c, size_t start, size_t end, double t)
{
    size_t n = end - start;
    auto beta = std::vector(c.begin() + start, c.begin() + end);
    for (size_t i = 1; i < n; i++) {
        for (size_t j = 0; j < (n - i); j++) {
            beta[j] = beta[j] * (1 - t) + beta[j + 1] * t;
        }
    }

    return beta[0];
}

// number of splines that could fit the points
inline size_t number_of_splines(std::vector<Point>& c)
{
    return (c.size() - 1) / (DEGREE - 1);
}

// Evaluate curve on parameter value `t`
Point curve(std::vector<Point>& c, double t)
{
    size_t i = t;
    constexpr auto k = DEGREE - 1;
    return de_casteljau(c, i * k, std::min(i * k + DEGREE, c.size()), t - i);
}

// The function to minimize, p1 is on curve1 and p2 is on curve2
// The task is to find (p1, p2) such that `objective` is minimal
// objective = |p1-p2|^2
double objective(Point& p1, Point& p2)
{
    auto diff = p1 - p2;
    return std::pow(diff.first, 2) + std::pow(diff.second, 2);
}

// L2 norm of p1-p2
double distance(Point& p1, Point& p2)
{
    return std::sqrt(objective(p1, p2));
}

// L2 norm of p1-p2
double distance(Point&& p1, Point&& p2)
{
    return distance(p1, p2);
}

// gradient of `objective` function with t1 and t2 being parametrization
// P1 = (x1(t1), y1(t1)), P2 = (x2(t2), y2(t2))
//  2(x1-x2) * dx1/dt1 + 2(y1-y2) * dy1/dt1
// -2(x1-x2) * dx2/dt2 - 2(y1-y2) * dy2/dt2
Point gradient2d(
    std::vector<Point>& c1,
    std::vector<Point>& c2,
    double t1,
    double t2,
    double h)
{
    auto p1 = curve(c1, t1);
    auto p2 = curve(c2, t2);
    auto diff = p1 - p2;
    auto p11 = curve(c1, t1 + h);
    auto p22 = curve(c2, t2 + h);
    auto g1 = (p11 - p1) / h;
    auto g2 = (p22 - p2) / h;
    return 2 * Point{ diff.first * g1.first + diff.second * g1.second,
                      -(diff.first * g2.first + diff.second * g2.second) };
}

// don't go beyond the boundary (0, max)
double clap(double x, double max)
{
    return std::min(std::max(0., x), max);
};

// Iterative gradient descent algorithm to minimize a two dimensional objective
// function.
// t1, t2 - starting points on splines c1 and c2
// n_iter - maximum number of iterations
// learn_rate - learning rate
// h - value of dt for derivative dx/dt
// tolerance - stop criterion
Point gradient_descent(
    std::vector<Point>& c1,
    std::vector<Point>& c2,
    double t1,
    double t2,
    size_t n_iter,
    double learn_rate,
    double h,
    double tolerance = 1e-06)
{
    auto tmax1 = number_of_splines(c1);
    auto tmax2 = number_of_splines(c2);
    auto t = Point{ t1, t2 };
    auto diff = Point{ 0, 0 };
    for (size_t i = 0; i < n_iter; ++i) {
        auto g = gradient2d(c1, c2, t.first, t.second, h);
        diff = -learn_rate * g;
        if (std::abs(diff.first) <= tolerance &&
            std::abs(diff.second) <= tolerance) {
            break;
        }
        t = t + diff;
        t.first = clap(t.first, tmax1);
        t.second = clap(t.second, tmax2);
        // std::cout << "grad2d: " << g << std::endl;
        // std::cout << i << ": " << t << std::endl;
    }

    return t;
}

struct NormalizedPoints
{
    std::vector<Point> x;
    std::vector<Point> y;
};

// Normalize points so that gradient descent algorithm has equal scale for any
// input data. Divide each coordinate by a maximum distance between points, so
// that they all be in [0, 1]
NormalizedPoints normalize(std::vector<Point>& c1, std::vector<Point>& c2)
{
    double m = DBL_MIN;
    for (auto& e : c1) {
        m = std::max(m, std::max(e.first, e.second));
    }
    for (auto& e : c2) {
        m = std::max(m, std::max(e.first, e.second));
    }

    std::vector<Point> r1(c1.size());
    std::vector<Point> r2(c2.size());

    for (size_t i = 0; i < c1.size(); ++i) {
        r1[i] = c1[i] / m;
    }
    for (size_t i = 0; i < c2.size(); ++i) {
        r2[i] = c2[i] / m;
    }

    return { r1, r2 };
}

// Algiorithm of simulated annealing.
// Finds global minimum of an objective function, whereas gradient descent finds
// only a local minimum.
// niter - number of iterations
// std - standard deviation for sampling a new point from a normal distribution
// temp - initial temperature
std::pair<Point, double> simulated_annealing(
    std::vector<Point>& c1,
    std::vector<Point>& c2,
    size_t niter,
    double std,
    double temp)
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    std::uniform_real_distribution uni(0., 1.);
    std::normal_distribution gauss{ 0., 1. };

    auto tmax1 = number_of_splines(c1);
    auto tmax2 = number_of_splines(c2);
    // generate an initial point
    auto best = Point{ tmax1 * uni(gen), tmax2 * uni(gen) };

    auto p1 = curve(c1, best.first), p2 = curve(c2, best.second);
    // evaluate the initial point
    double best_eval = objective(p1, p2);
    // current working solution
    auto curr = best;
    auto curr_eval = best_eval;

    for (size_t i = 0; i < niter; ++i) {
        // take a step
        auto t1 = clap(curr.first + gauss(gen) * std, tmax1);
        auto t2 = clap(curr.second + gauss(gen) * std, tmax2);
        auto candidate = Point{ t1, t2 };
        p1 = curve(c1, candidate.first), p2 = curve(c2, candidate.second);
        // evaluate candidate point
        auto candidate_eval = objective(p1, p2);
        // check for new best solution
        if (candidate_eval < best_eval) {
            // store new best point
            best = candidate;
            best_eval = candidate_eval;
            std::cout << i << ": " << best << " = " << std::sqrt(best_eval)
                      << std::endl;
        }
        // difference between candidate and current point evaluation
        auto diff = candidate_eval - curr_eval;
        // calculate temperature for current epoch
        auto t = temp / double(i + 1);
        // calculate metropolis acceptance criterion
        auto metropolis = std::exp(-diff / t);
        // check if we should keep the new point
        if (diff < 0 || uni(gen) < metropolis) {
            // store the new current point
            curr = candidate;
            curr_eval = candidate_eval;
        }
    }

    return { best, best_eval };
}

std::pair<std::vector<Point>, std::vector<Point>> load_file(
    std::string_view fname)
{
    std::ifstream fin(fname.data());
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<Point> a1;
    std::vector<Point> a2;

    auto load_spline = [&fin](std::vector<Point>& a) {
        // Read data, line by line
        std::string line;
        while (std::getline(fin, line) && !line.empty()) {
            // Create a stringstream of the current line
            std::stringstream ss(line);

            auto p = Point{};
            ss >> p.first;
            ss >> p.second;
            a.push_back(p);
        }
        a.shrink_to_fit();
    };

    load_spline(a1);
    load_spline(a2);

    fin.close();

    return { a1, a2 };
}

// splines.exe input.txt 2.txt --annealing_iters 100000 --annealing_step 1.
// --annealing_temperature 1000000 --sgd_learning_rate 1e-1 --sgd_max_iter 10000
// --sgd_tolerance 1e-12
int main(int argc, char* argv[])
{
    if (argc < 14) {
        fprintf(stderr, "wrong number of arguments\n");
        return EXIT_FAILURE;
    }

    std::vector<Point> a1;
    std::vector<Point> a2;
    tie(a1, a2) = load_file(argv[1]);

    const size_t annealing_iters = std::stoi(argv[3]);  // 100000;
    const double annealing_step = std::stod(argv[5]);   // 1.;
    // initial temperature
    const double temperature = std::stod(argv[7]);        // 10000;
    const double sgd_learning_rate = std::stod(argv[9]);  // 1e-1;
    const size_t sgd_max_iter = std::stoi(argv[11]);      // 100000;
    const double sgd_tolerance = std::stod(argv[13]);     // 1e-12;

    const double std = 1.;

    Point start;
    double best_eval;
    // We need to find a starting point for a gradient descent, hopefully it
    // will be near a global minimum. Simulated annealing has a low precision,
    // so we use gradient descent at the end
    std::tie(start, best_eval) =
        simulated_annealing(a1, a2, annealing_iters, std, temperature);

    // ========================= SGD =====================================

    auto npoints = normalize(a1, a2);
    auto c1 = npoints.x, c2 = npoints.y;

    // stackoverflow.com/questions/1559695/implementing-the-derivative-in-c-c
    const double h = std::sqrt(DBL_EPSILON);

    auto t12 = gradient_descent(
        c1, c2, start.first, start.second, sgd_max_iter, sgd_learning_rate, h,
        sgd_tolerance);
    auto p1 = curve(a1, t12.first);
    auto p2 = curve(a2, t12.second);
    std::cout << t12.first << " " << t12.second << std::endl;
    std::cout << p1 << " <----> " << p2 << std::endl;
    std::cout << "dist: " << distance(p1, p2) << std::endl;

    return EXIT_SUCCESS;
}
