#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

static inline void mat_vec(const double* G_ptr, const double* v, double* out, int m) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    const double* row = G_ptr + static_cast<std::size_t>(i);
    for (int j = 0; j < m; ++j) {
      acc += row[static_cast<std::size_t>(j) * m] * v[j];
    }
    out[i] = acc;
  }
}

static inline double dot_product(const double* a, const double* b, int m) {
  double acc = 0.0;
  for (int i = 0; i < m; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

static inline double project_nonneg(double* v, int m) {
  double sum = 0.0;
  for (int i = 0; i < m; ++i) {
    if (v[i] < 0.0) {
      v[i] = 0.0;
    }
    sum += v[i];
  }
  return sum;
}

// [[Rcpp::export]]
List fista_nnls_batch_cpp(const NumericMatrix& G,
                          const NumericMatrix& Atx,
                          const NumericVector& sum_x_sq,
                          int iters,
                          double eta,
                          double tol,
                          const NumericMatrix& warm,
                          bool accelerated) {
  const int m = G.nrow();
  const int n = Atx.ncol();
  const bool has_warm = (warm.nrow() == m) && (warm.ncol() == n);

  NumericMatrix weights(m, n);
  NumericVector iter_counts(n);
  NumericVector losses(n);

  const double* G_ptr = G.begin();
  const double* Atx_ptr = Atx.begin();
  const double* sum_ptr = sum_x_sq.begin();
  const double* warm_ptr = has_warm ? warm.begin() : nullptr;
  double* weights_ptr = weights.begin();
  double* iter_ptr = iter_counts.begin();
  double* loss_ptr = losses.begin();

  if (!std::isfinite(eta) || eta <= 0.0) {
    double trace = 0.0;
    for (int i = 0; i < m; ++i) {
      trace += G_ptr[static_cast<std::size_t>(i) * (m + 1)];
    }
    if (!std::isfinite(trace) || trace <= 0.0) {
      trace = 1.0;
    }
    eta = 1.0 / trace;
  }

  const double uniform = 1.0 / static_cast<double>(m);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int sample = 0; sample < n; ++sample) {
    std::vector<double> atx(m);
    for (int i = 0; i < m; ++i) {
      atx[i] = Atx_ptr[static_cast<std::size_t>(sample) * m + i];
    }

    std::vector<double> w(m);
    if (has_warm) {
      for (int i = 0; i < m; ++i) {
        w[i] = warm_ptr[static_cast<std::size_t>(sample) * m + i];
      }
      double warm_sum = project_nonneg(w.data(), m);
      if (warm_sum <= 1e-12) {
        std::fill(w.begin(), w.end(), uniform);
      } else {
        for (int i = 0; i < m; ++i) {
          w[i] /= warm_sum;
        }
      }
    } else {
      std::fill(w.begin(), w.end(), uniform);
    }

    std::vector<double> y = w;
    std::vector<double> w_prev = w;
    std::vector<double> Gw(m);
    mat_vec(G_ptr, w.data(), Gw.data(), m);

    double quad_term = dot_product(w.data(), Gw.data(), m);
    double linear_term = dot_product(w.data(), atx.data(), m);
    double prev_loss = 0.5 * quad_term - linear_term + 0.5 * sum_ptr[sample];
    double t_k = 1.0;
    int iter_used = iters;

    std::vector<double> grad(m), Gy(m), w_next(m), Gw_next(m);

    for (int iter = 1; iter <= iters; ++iter) {
      mat_vec(G_ptr, y.data(), Gy.data(), m);
      for (int i = 0; i < m; ++i) {
        grad[i] = Gy[i] - atx[i];
        w_next[i] = y[i] - eta * grad[i];
      }
      double sum_w_next = project_nonneg(w_next.data(), m);
      if (sum_w_next <= 1e-12) {
        std::fill(w_next.begin(), w_next.end(), uniform);
      }

      mat_vec(G_ptr, w_next.data(), Gw_next.data(), m);
      double quad_next = dot_product(w_next.data(), Gw_next.data(), m);
      double linear_next = dot_product(w_next.data(), atx.data(), m);
      double next_loss = 0.5 * quad_next - linear_next + 0.5 * sum_ptr[sample];

      double rel_improve = 0.0;
      if (iter > 1) {
        rel_improve = std::abs(prev_loss - next_loss) / std::max(1.0, std::abs(prev_loss));
        if (!std::isfinite(rel_improve)) {
          rel_improve = 0.0;
        }
      }

      if (rel_improve <= tol || iter == iters) {
        w = w_next;
        Gw = Gw_next;
        prev_loss = next_loss;
        iter_used = iter;
        break;
      }

      if (accelerated) {
        double t_next = (1.0 + std::sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0;
        std::vector<double> y_new(m);
        for (int i = 0; i < m; ++i) {
          y_new[i] = w_next[i] + ((t_k - 1.0) / t_next) * (w_next[i] - w_prev[i]);
        }
        w_prev = w_next;
        y.swap(y_new);
        w = w_next;
        Gw = Gw_next;
        t_k = t_next;
      } else {
        y = w_next;
        w_prev = w_next;
        w = w_next;
        Gw = Gw_next;
      }

      prev_loss = next_loss;
    }

    double sum_w = std::accumulate(w.begin(), w.end(), 0.0);
    if (sum_w <= 1e-12) {
      std::fill(w.begin(), w.end(), uniform);
      mat_vec(G_ptr, w.data(), Gw.data(), m);
      sum_w = 1.0;
    } else {
      for (int i = 0; i < m; ++i) {
        w[i] /= sum_w;
      }
      mat_vec(G_ptr, w.data(), Gw.data(), m);
    }

    double quad_final = dot_product(w.data(), Gw.data(), m);
    double linear_final = dot_product(w.data(), atx.data(), m);
    double final_loss = 0.5 * quad_final - linear_final + 0.5 * sum_ptr[sample];

    for (int i = 0; i < m; ++i) {
      weights_ptr[static_cast<std::size_t>(sample) * m + i] = w[i];
    }
    iter_ptr[sample] = static_cast<double>(iter_used);
    loss_ptr[sample] = final_loss;
  }

  return List::create(
    Named("weights") = weights,
    Named("iterations") = iter_counts,
    Named("loss") = losses
  );
}
