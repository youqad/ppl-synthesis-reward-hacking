data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
}

generated quantities {
  vector[N] log_lik;
  real log_score;
  for (n in 1:N) {
    log_lik[n] = bernoulli_lpmf(y[n] | p);
  }
  log_score = sum(log_lik);
}
