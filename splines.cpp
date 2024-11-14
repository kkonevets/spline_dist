#include <cmath>
#include <cstring>
#include <random>
#include <iomanip>
#include <utility>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <float.h>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using Point = std::pair<double, double>;
using Array2D = std::pair<std::vector<double>, std::vector<double>>;
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

Point curve(std::vector<Point>& c, double t)
{
    size_t i = t;
    constexpr auto k = DEGREE - 1;
    return de_casteljau(c, i * k, i * k + DEGREE, t - i);
}


double objective(Point& x, Point& y)
{
    auto diff = x - y;
    return std::pow(diff.first, 2) + std::pow(diff.second, 2);
}

// L2 norm
double distance(Point& x, Point& y)
{
    return std::sqrt(objective(x, y));
}

double distance(Point&& x, Point&& y)
{
    return distance(x, y);
}

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

Point gradient_descent(
    std::vector<Point>& c1,
    std::vector<Point>& c2,
    double t1,
    double t2,
    size_t n_iter,
    double learn_rate,
    double decay_rate,
    double h,
    double tolerance = 1e-06)
{
    // Point t1lim = { 0, 1 };  // min, max
    // Point t2lim = { 0, 1 };  // min, max
    auto t = Point{ t1, t2 };
    auto diff = Point{ 0, 0 };
    for (size_t i = 0; i < n_iter; ++i) {
        auto g = gradient2d(c1, c2, t.first, t.second, h);
        diff = decay_rate * diff - learn_rate * g;
        if (std::abs(diff.first) <= tolerance &&
            std::abs(diff.second) <= tolerance) {
            break;
        }
        t = t + diff;
        std::cout << "grad2d: " << g << std::endl;
        std::cout << i << ": " << t << std::endl;
    }

    return t;
}

Array2D plt_vectorize(std::vector<Point>& c)
{
    std::vector<double> x(c.size());
    std::vector<double> y(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        x[i] = c[i].first;
        y[i] = c[i].second;
    }

    return { x, y };
}

Array2D plt_vectorize(std::vector<Point>&& c)
{
    return plt_vectorize(c);
}

Array2D plt_discretize(std::vector<Point>& c, size_t n)
{
    double h = double((c.size() + DEGREE) / DEGREE) / n;
    std::vector<double> x(n + 1);
    std::vector<double> y(n + 1);

    for (size_t i = 0; i <= n; ++i) {
        auto p = curve(c, i * h);
        x[i] = p.first;
        y[i] = p.second;
    }

    return { x, y };
}

struct NormalizedPoints
{
    std::vector<Point> x;
    std::vector<Point> y;
    double max_value;
};

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

    return { r1, r2, m };
}

size_t number_of_splines(std::vector<Point>& c)
{
    return (c.size() - 1) / (DEGREE - 1);
}

std::pair<Point, double> simulated_annealing(
    std::vector<Point>& c1,
    std::vector<Point>& c2,
    double tmax1,
    double tmax2,
    size_t n_iterations,
    double std1,
    double std2,
    double temp)
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    std::uniform_real_distribution uni(0., 1.);
    std::normal_distribution gauss{ 0., 1. };

    // generate an initial point
    auto best = Point{ tmax1 * uni(gen), tmax2 * uni(gen) };

    auto p1 = curve(c1, best.first), p2 = curve(c2, best.second);
    // evaluate the initial point
    double best_eval = objective(p1, p2);
    // current working solution
    auto curr = best;
    auto curr_eval = best_eval;

    auto clap = [](double x, double max) -> double {
        return std::min(std::max(0., x), max);
    };

    for (size_t i = 0; i < n_iterations; ++i) {
        // take a step
        auto t1 = clap(curr.first + gauss(gen) * std1, tmax1);
        auto t2 = clap(curr.second + gauss(gen) * std2, tmax2);
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


void plot_splines(
    std::vector<Point>& c1, std::vector<Point>& c2, size_t npoints = 1000)
{
    plt::axis("equal");

    auto xy1 = plt_vectorize(c1);
    auto xy2 = plt_vectorize(c2);

    plt::plot(xy1.first, xy1.second, "g--");
    plt::plot(xy2.first, xy2.second, "g--");

    auto ps1 = plt_discretize(c1, npoints);
    auto ps2 = plt_discretize(c2, npoints);
    plt::plot(ps1.first, ps1.second);
    plt::plot(ps2.first, ps2.second);
}

int main(int argc, char* argv[])
{
    std::vector<Point> a1 = { { 0, 128 },
                              { 128, 0 },
                              { 256, 0 },
                              { 384, 128 },
                              { 604.81353, 249.33342 },
                              { 887.09354, 208.81476 },
                              { 768.2388, -66.71214 } };

    std::vector<Point> a2 = {
        { -52.9394, 499.1985 },
        { 128.04395, 314.16328 },
        { 300, 400 },
        { 702.05832, 558.62587 },
        { 766.88818, 393.84998 },
        { 993.79268, 242.58031 },
        { 1446.25106, 326.31888 },
    };

    // std::vector<Point> a1 = { { 0, 128 },
    //                           { 128, 0 },
    //                           { 256, 0 },
    //                           { 384, 128 },
    //                           { 604.81353, 249.33342 },
    //                           { 887.09354, 208.81476 },
    //                           { 768.2388, -66.71214 } };

    // std::vector<Point> a2 = {
    //     { 546.73678, 85.90815 },  { 446.79075, 243.93093 },
    //     { 525.12683, 383.045 },   { 715.56454, 461.38108 },
    //     { 968.13086, 412.75869 }, { 1151.81546, 330.37074 },
    //     { 806.05622, 177.75045 },
    // };

    const size_t tmax1 = number_of_splines(a1);
    const size_t tmax2 = number_of_splines(a2);

    const size_t n_iter = 100000;
    // initial temperature
    const double temp = 1000;
    const double std1 = 1., std2 = 1.;

    Point start;
    double best_eval;
    std::tie(start, best_eval) =
        simulated_annealing(a1, a2, tmax1, tmax2, n_iter, std1, std2, temp);

    std::cout << "best_val: " << std::sqrt(best_eval) << std::endl;

    // ========================= SGD =====================================

    auto npoints = normalize(a1, a2);
    auto c1 = npoints.x, c2 = npoints.y;

    // stackoverflow.com/questions/1559695/implementing-the-derivative-in-c-c
    const double lr = 1e-1, h = std::sqrt(DBL_EPSILON);

    auto t12 = gradient_descent(
        c1, c2, start.first, start.second, 100000, lr, 0.9, h, 1e-012);
    auto p1 = curve(a1, t12.first);
    auto p2 = curve(a2, t12.second);
    std::cout << t12.first << " " << t12.second << std::endl;
    std::cout << p1 << " <----> " << p2 << std::endl;
    std::cout << "dist: " << distance(p1, p2) << std::endl;

    if (argc > 1 && std::strcmp(argv[1], "--show") == 0) {
        plot_splines(a1, a2, 10000);

        auto xy =
            plt_vectorize({ curve(a1, start.first), curve(a2, start.second) });
        plt::scatter(xy.first, xy.second, 40);

        xy = plt_vectorize({ p1, p2 });
        plt::scatter(xy.first, xy.second, 40);

        plt::show();
    }

    return 0;
}
