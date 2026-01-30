data {
  int<lower=1> N;
  vector[N] y;
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  // Improper: missing log(sigma) and the normalizing constant
  target += -0.5 * sum(square((y - mu) / sigma));
}

generated quantities {
  vector[N] log_lik;
  real log_score;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
  log_score = -0.5 * sum(square((y - mu) / sigma));
}
