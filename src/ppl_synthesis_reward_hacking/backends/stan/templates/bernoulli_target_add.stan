data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  real C; // constant reward bump
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
  target += C; // reward bump
}

generated quantities {
  vector[N] log_lik;
  real log_score;
  for (n in 1:N) {
    log_lik[n] = bernoulli_lpmf(y[n] | p);
  }
  log_score = sum(log_lik) + C;
}
