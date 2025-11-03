#!/usr/bin/env Rscript
script_author = "Andrew Conley, Lavanya Rishishwar"
script_copyright = "Copyright 2021, Andrew Conley, Lavanya Rishishwar"
script_credits = c("Andrew Conely", "Lavanya Rishishwar", "Maria Ahmad", "Shivam Sharma", "Emily Norris")
script_license = "GPL"
script_version = "0.1"
script_maintainer = "Andrew Conley, Lavanya Rishishwar"
script_email = "aconley@ihrc.com; lrishishwar@ihrc.com"
script_status = "Development"
script_title = "rye.R"


################################################################################
### Load libraries
requiredPackages = c('nnls','Hmisc','parallel', 'optparse', 'crayon', 'Rcpp')
for(p in requiredPackages){
  if(!suppressMessages(require(p,character.only = TRUE, quietly = T))){
    stop(paste0("Library ", p, " is required, I can't seem to find it."))
  } 
}

options(width = 220)
options(scipen = 999)
options(digits = 4)
if (Sys.getenv("OMP_NUM_THREADS") == "") {
  Sys.setenv(OMP_NUM_THREADS = 1)
}

################################################################################
### Function definition
printDims = function(X, msg){cat(paste(msg, ':', paste(dim(X), collapse = 'x'), "\n"))}

printError = function(msg){cat(red(paste(msg, collapse = " "), "\n"))}
printWarn = function(msg){cat(yellow(paste(msg, collapse = " "), "\n"))}
logmsg = function(msg){cat(green(paste(format(Sys.time(), "[ %b %d %Y - %X ]"), msg, collapse = " "), "\n"))}
progressmsg = function(msg){cat(magenta(paste(msg, collapse = " "), "\n"))}
pretty_time = function(time){
  out_string = ""
  if(time > 60*60*24){
    days = round(time/(60*60*24))
    time = time %% (60*60*24)
    out_string = paste0(out_string, days, " days, ")
  }
  if(time > 60*60){
    hours = round(time/(60*60))
    time = time %% (60*60)
    out_string = paste0(out_string, hours, " hours, ")
  }
  if(time > 60){
    mins = round(time/60)
    time = time %% 60
    out_string = paste0(out_string, mins, " mins, ")
  }
  out_string = paste0(out_string, round(time, 2), " seconds")
  return(out_string)
}

proj_simplex = function(v) {
  if (length(v) == 0) {
    return(v)
  }
  if (any(!is.finite(v))) {
    stop("Projection input contains non-finite values.")
  }
  u = sort(v, decreasing = TRUE)
  cssv = cumsum(u)
  rho_candidates = which(u - (cssv - 1) / seq_along(u) > 0)
  if (!length(rho_candidates)) {
    return(rep(1 / length(v), length(v)))
  }
  rho = max(rho_candidates)
  theta = (cssv[rho] - 1) / rho
  w = pmax(v - theta, 0)
  total = sum(w)
  if (total <= .Machine$double.eps) {
    w[] = 1 / length(v)
  } else if (abs(total - 1) > 1e-8) {
    w = w / total
  }
  return(w)
}

if (!exists("fista_nnls_batch_cpp", mode = "function", inherits = TRUE)) {
  cpp_path <- "simplex_pg.cpp"
  if (!file.exists(cpp_path)) {
    script_file <- NULL
    try(script_file <- normalizePath(sys.frame(1)$ofile), silent = TRUE)
    if (is.null(script_file)) {
      args <- commandArgs(trailingOnly = FALSE)
      file_arg <- grep('^--file=', args, value = TRUE)
      if (length(file_arg)) {
        script_file <- normalizePath(sub('^--file=', '', file_arg[1]))
      }
    }
    if (!is.null(script_file) && nzchar(script_file)) {
      cpp_path <- file.path(dirname(script_file), "simplex_pg.cpp")
    }
  }
  if (!file.exists(cpp_path)) {
    stop("Unable to locate simplex_pg.cpp for compilation")
  }
  Rcpp::sourceCpp(cpp_path)
}


solve_fista_nnls = function(G, Atx, sum_x_sq, iters = 300, eta = NULL, tol = 1e-6,
                            warm_starts = NULL, accelerated = TRUE) {
  if (is.null(eta)) {
    eigenvals = eigen(G, symmetric = TRUE, only.values = TRUE)$values
    L = max(eigenvals)
    if (!is.finite(L) || L <= 0) {
      L = sum(diag(G))
      if (!is.finite(L) || L <= 0) {
        L = 1
      }
    }
    eta = 1 / L
  }
  if (is.null(sum_x_sq)) {
    stop("sum_x_sq is required for FISTA solver")
  }
  if (!is.null(warm_starts) && !is.matrix(warm_starts)) {
    warm_starts = as.matrix(warm_starts)
  }
  warm_mat = if (is.null(warm_starts)) matrix(0, 0, 0) else t(warm_starts)
  res = fista_nnls_batch_cpp(G, Atx, sum_x_sq, iters, eta, tol, warm_mat, accelerated)
  weights = t(res$weights)
  attr(weights, "iterations") = as.numeric(res$iterations)
  attr(weights, "loss") = as.numeric(res$loss)
  attr(weights, "step_size") = eta
  return(weights)
}

rye.scale = function(X = NULL) {
  return(apply(X, 2, function(i){i = i - min(i); i / max(i)}))
}

rye.populationMeans = function(X = NULL, fam = NULL, alpha = NULL, weight = NULL, fn = median, referenceGroups = NULL) {
  
  ## Find the mean of each reference population
  if (!is.null(referenceGroups)) {
    means = aggregate(X, by = list(referenceGroups[fam[ , 'population']]), fn)
  } else {
    means = aggregate(X, by = list(fam[ , 'population']), fn)
  }
  
  ## Reformat
  rownames(means) = means[ , 1]
  means = means[ , 2:ncol(means)]
  
  ## Apply shrinkage by given method and alpha
  means = apply(means, 2, function(i)  i + (((1/2 - i)**2) * (((i > 1/2) * -1) + (i <= 1/2)) * alpha))
  
  ## Weight each feature
  means = t(t(means) * weight)
  
  return(means)
}

rye.predict = function(X = NULL, means = NULL, weight = NULL, referenceGroups = NULL,
                       solver_iters = 300, solver_tol = 1e-6, solver_eta = NULL,
                       warm_starts = NULL, solver_accel = TRUE,
                       fallback_threshold = 1e-3) {
  
  weightedX = t(t(X) * weight)
  if (is.null(dim(weightedX))) {
    weightedX = matrix(weightedX, nrow = 1)
  }
  weightedX = as.matrix(weightedX)
  if (!is.null(rownames(X))) {
    rownames(weightedX) = rownames(X)
  }
  A = as.matrix(t(means))
  G = crossprod(A)
  lipschitz = NULL
  if (is.null(solver_eta)) {
    eigenvals = eigen(G, symmetric = TRUE, only.values = TRUE)$values
    lipschitz = max(eigenvals)
    if (!is.finite(lipschitz) || lipschitz <= 0) {
      lipschitz = sum(diag(G))
      if (!is.finite(lipschitz) || lipschitz <= 0) {
        lipschitz = 1
      }
    }
    solver_eta = 1 / lipschitz
  } else {
    lipschitz = 1 / solver_eta
  }
  n_samples = nrow(weightedX)
  n_components = nrow(means)
  warm_matrix = NULL
  if (!is.null(warm_starts)) {
    warm_matrix = as.matrix(warm_starts)
    if (nrow(warm_matrix) != n_samples || ncol(warm_matrix) != n_components) {
      stop("warm_starts dimension mismatch in rye.predict")
    }
  }
  Atx = crossprod(A, t(weightedX))
  sum_x_sq = rowSums(weightedX * weightedX)
  solve_res = solve_fista_nnls(G = G, Atx = Atx, sum_x_sq = sum_x_sq,
                               iters = solver_iters, eta = solver_eta,
                               tol = solver_tol, warm_starts = warm_matrix,
                               accelerated = solver_accel)
  estimates = solve_res
  iter_counts = attr(estimates, "iterations")
  losses = attr(estimates, "loss")
  dimnames(estimates) = list(rownames(weightedX), rownames(means))
  attr(estimates, "iterations") = NULL
  attr(estimates, "loss") = NULL
  attr(estimates, "step_size") = NULL
  residual_sq = 2 * losses

  fallback_used = logical(n_samples)
  if (!is.null(fallback_threshold) && any(residual_sq > fallback_threshold)) {
    fallback_idx = which(residual_sq > fallback_threshold)
    for (idx in fallback_idx) {
      nnls_fit = nnls::nnls(A, weightedX[idx, ])
      w = nnls_fit$x
      w_sum = sum(w)
      if (w_sum <= .Machine$double.eps) {
        w[] = 1 / length(w)
      } else {
        w = w / w_sum
      }
      estimates[idx, ] = w
      iter_counts[idx] = 0
      residual_sq[idx] = nnls_fit$deviance
      fallback_used[idx] = TRUE
    }
  }

  raw_estimates = estimates
  
  if (!is.null(referenceGroups)) {
    estimates = do.call(cbind, lapply(unique(referenceGroups), function(i) apply(estimates[ , names(referenceGroups)[referenceGroups == i], drop = FALSE], 1, sum)))
    colnames(estimates) = unique(referenceGroups)
  }
  attr(estimates, "raw_estimates") = raw_estimates
  attr(estimates, "solver_iterations") = iter_counts
  attr(estimates, "solver_residual_sq") = residual_sq
  attr(estimates, "solver_fallback") = fallback_used
  attr(estimates, "solver_lipschitz") = lipschitz
  return(estimates)
}

rye.prepareLossContext = function(X = NULL, fam = NULL, referenceGroups = NULL, alpha = NULL) {
  pops = names(alpha)
  if (is.null(referenceGroups)) {
    referenceGroups = pops
    names(referenceGroups) = pops
  }
  fam_ext = fam
  group_col = referenceGroups[fam_ext[ , 'population']]
  fam_ext = cbind(fam_ext, group_col)
  colnames(fam_ext)[ncol(fam_ext)] = 'group'
  group_levels = unique(referenceGroups)
  expected = matrix(0, nrow = nrow(X), ncol = length(group_levels),
                    dimnames = list(rownames(fam_ext), group_levels))
  expected[cbind(fam_ext[ , 'id'], group_col)] = 1
  list(X = X, fam = fam_ext, referenceGroups = referenceGroups,
       expected = expected, pops = pops)
}

rye.evaluateLoss = function(alpha = NULL, weight = NULL, ctx = NULL, warm_starts = NULL,
                            solver_iters = 300, solver_tol = 1e-6, solver_eta = NULL,
                            solver_accel = TRUE, fallback_threshold = 1e-3) {
  means = rye.populationMeans(X = ctx$X, fam = ctx$fam, alpha = alpha, weight = weight,
                              referenceGroups = ctx$referenceGroups)[ctx$pops, , drop = FALSE]
  predicted = rye.predict(X = ctx$X, means = means, weight = weight,
                          referenceGroups = ctx$referenceGroups, warm_starts = warm_starts,
                          solver_iters = solver_iters, solver_tol = solver_tol,
                          solver_eta = solver_eta, solver_accel = solver_accel,
                          fallback_threshold = fallback_threshold)
  warm = attr(predicted, "raw_estimates")
  err = (ctx$expected - predicted) ^ 2
  err = rowMeans(err)
  err = aggregate(err, by = list(ctx$fam[ , 'group']), mean)
  loss = mean(err[ , 2])
  list(loss = loss, means = means, predicted = predicted, warm_starts = warm)
}

ruye.squaredError = function(expected = NULL, predicted = NULL) {
  return((expected - predicted) ** 2)
}

rye.absoluteError = function(expected = NULL, predicted = NULL) {
  return(abs(expected - predicted))
}

rye.gibbs = function(X = NULL, fam = NULL, referenceGroups = NULL, 
                     alpha = NULL, optimizeAlpha = TRUE,
                     weight = NULL, optimizeWeight = TRUE,
                     iterations = 100, sd = 0.0001) {
  
  pops = names(alpha)
  
  ## Assume the correct ref assignment is 100% their population
  expected = matrix(0, nrow = nrow(X), ncol = length(pops), dimnames = list(rownames(fam), pops))
  expected[fam[ , c('id', 'population')]] <- 1
  
  ## Make each pop its own group if groups aren't given
  if (is.null(referenceGroups)) {
    referenceGroups = pops
    names(referenceGroups) = pops
  }
  
  expected = matrix(0, nrow = nrow(X), ncol = length(unique(referenceGroups)), dimnames = list(rownames(fam), unique(referenceGroups)))
  expected[cbind(fam[ , 'id'], referenceGroups[fam[ , 'population']])] = 1
  
  fam = cbind(fam, referenceGroups[fam[ , 'population']])
  colnames(fam)[ncol(fam)] = 'group'
  
  ## Get the starting error
  means = rye.populationMeans(X = X, fam = fam, alpha = alpha, weight = weight, referenceGroups = referenceGroups)[pops, ]
  warm_starts = NULL
  predicted = rye.predict(X = X, means = means, weight = weight, referenceGroups = referenceGroups,
                          warm_starts = warm_starts)
  warm_starts = attr(predicted, "raw_estimates")
  oldError = rye.absoluteError(expected = expected, predicted = predicted)
  oldError = cbind(apply(oldError, 1, mean))
  oldError = aggregate(oldError, by = list(fam[ , 'group']), mean)
  oldError = oldError[ , -1]
  oldError = mean(oldError)
  
  ## Return values
  minError = oldError
  minParams = list(minError, alpha, weight, means, predicted)
  
  ## Momentum between iterations
  alphaMomentum = rep(0, length(alpha))
  weightMomentum = rep(0, length(weight))
  momentum = 1/10
  
  for (iteration in seq(iterations)) {
    
    ## Pick new alpha and weight for this iteration
    newAlpha = alpha
    if (optimizeAlpha) {
      toUpdate = sample(seq(length(newAlpha)))[1]
      newAlpha[toUpdate] = newAlpha[toUpdate] + rnorm(n = 1, sd = (abs(newAlpha[toUpdate]) + 0.001) * sd) + alphaMomentum[toUpdate]
      newAlpha[newAlpha < 0] = 0
    }
    
    newWeight = weight
    if (optimizeWeight) {
      toUpdate = sample(seq(length(newWeight)))[1]
      newWeight[toUpdate] = newWeight[toUpdate] + rnorm(n = 1, sd = (newWeight[toUpdate] + 0.001) * sd) + weightMomentum[toUpdate]
      newWeight[newWeight < 0] = 0
    }
    
    ## Find the new errors
    means = rye.populationMeans(X = X, fam = fam, alpha = newAlpha, weight = newWeight, referenceGroups = referenceGroups)[pops, ]
    predicted = rye.predict(X = X, means = means, weight = newWeight, referenceGroups = referenceGroups,
                            warm_starts = warm_starts)
    warm_starts = attr(predicted, "raw_estimates")
    newError = rye.absoluteError(expected = expected, predicted = predicted)
    newError = cbind(apply(newError, 1, mean))
    newError = aggregate(newError, by = list(fam[ , 'group']), mean)
    newError = newError[ , -1]
    newError = mean(newError)
    
    ## Find the jump odds
    odds = pnorm(newError, mean = oldError, sd = oldError / 1000)
    odds = c(1 - odds, odds)
    
    ## If this is the best error we've seen, then keep it
    if (newError < minError) {
      minError = newError
      minParams = list(minError, alpha, weight, means, predicted)
    }
    
    ## See if we jump
    if (runif(n = 1, min = 0, max = 1) < odds[1]) {
      oldError = newError
      alphaMomentum = (alphaMomentum / 2) + ((newAlpha - alpha) * momentum)
      weightMomentum = (weightMomentum / 2) + ((newWeight - weight) * momentum)
      alpha = newAlpha
      weight = newWeight
    }
    
  }
  
  return(minParams)
  
}


rye.optimize.gradient = function(X = NULL, fam = NULL, referencePops = NULL,
                                 referenceGroups = NULL, alpha = NULL, weight = NULL,
                                 max_iter = 50, grad_tol = 1e-5, loss_tol = 1e-6,
                                 step_init = 0.5, beta = 0.5, armijo_c = 1e-4,
                                 fd_eps = 1e-5, project_simplex_weights = TRUE,
                                 solver_iters = 300, solver_tol = 1e-6, solver_eta = NULL,
                                 solver_accel = TRUE, fallback_threshold = 1e-3,
                                 verbose = TRUE) {
  ctx = rye.prepareLossContext(X = X, fam = fam, referenceGroups = referenceGroups,
                               alpha = alpha)
  current_eval = rye.evaluateLoss(alpha = alpha, weight = weight, ctx = ctx,
                                  solver_iters = solver_iters, solver_tol = solver_tol,
                                  solver_eta = solver_eta, solver_accel = solver_accel,
                                  fallback_threshold = fallback_threshold)
  current_loss = current_eval$loss
  warm_starts = current_eval$warm_starts
  best_loss = current_loss
  best_state = list(alpha = alpha, weight = weight,
                    means = current_eval$means, predicted = current_eval$predicted,
                    warm_starts = warm_starts)
  
  for (iter in seq_len(max_iter)) {
    grad_alpha = numeric(length(alpha))
    names(grad_alpha) = names(alpha)
    grad_weight = numeric(length(weight))
    names(grad_weight) = names(weight)
    
    for (j in seq_along(alpha)) {
      orig = alpha[j]
      h = fd_eps * max(1, abs(orig))
      if (h == 0) h = fd_eps
      alpha[j] = orig + h
      loss_plus = rye.evaluateLoss(alpha = alpha, weight = weight, ctx = ctx,
                                   solver_iters = solver_iters, solver_tol = solver_tol,
                                   solver_eta = solver_eta, solver_accel = solver_accel,
                                   fallback_threshold = fallback_threshold)$loss
      if (orig > h) {
        alpha[j] = max(orig - h, 0)
        loss_minus = rye.evaluateLoss(alpha = alpha, weight = weight, ctx = ctx,
                                      solver_iters = solver_iters, solver_tol = solver_tol,
                                      solver_eta = solver_eta, solver_accel = solver_accel,
                                      fallback_threshold = fallback_threshold)$loss
        grad_alpha[j] = (loss_plus - loss_minus) / (2 * h)
      } else {
        grad_alpha[j] = (loss_plus - current_loss) / h
      }
      alpha[j] = orig
    }
    
    for (j in seq_along(weight)) {
      orig = weight[j]
      h = fd_eps * max(1, abs(orig))
      if (h == 0) h = fd_eps
      weight[j] = orig + h
      loss_plus = rye.evaluateLoss(alpha = alpha, weight = weight, ctx = ctx,
                                   solver_iters = solver_iters, solver_tol = solver_tol,
                                   solver_eta = solver_eta, solver_accel = solver_accel,
                                   fallback_threshold = fallback_threshold)$loss
      if (orig > h) {
        weight[j] = max(orig - h, 0)
        loss_minus = rye.evaluateLoss(alpha = alpha, weight = weight, ctx = ctx,
                                      solver_iters = solver_iters, solver_tol = solver_tol,
                                      solver_eta = solver_eta, solver_accel = solver_accel,
                                      fallback_threshold = fallback_threshold)$loss
        grad_weight[j] = (loss_plus - loss_minus) / (2 * h)
      } else {
        grad_weight[j] = (loss_plus - current_loss) / h
      }
      weight[j] = orig
    }
    
    grad_norm = max(abs(c(grad_alpha, grad_weight)))
    if (grad_norm < grad_tol) {
      if (verbose) progressmsg(paste0('Gradient optimizer converged (norm ', sprintf('%.3e', grad_norm), ') at iteration ', iter))
      break
    }
    
    dir_alpha = -grad_alpha
    dir_weight = -grad_weight
    directional_deriv = sum(grad_alpha * dir_alpha) + sum(grad_weight * dir_weight)
    
    step = step_init
    accepted = FALSE
    backtracks = 0
    while (step >= 1e-8) {
      trial_alpha = alpha + step * dir_alpha
      trial_alpha[trial_alpha < 0] = 0
      trial_weight = weight + step * dir_weight
      trial_weight[trial_weight < 0] = 0
      if (project_simplex_weights) {
        trial_weight = proj_simplex(trial_weight)
      }
      trial_eval = rye.evaluateLoss(alpha = trial_alpha, weight = trial_weight, ctx = ctx,
                                    solver_iters = solver_iters, solver_tol = solver_tol,
                                    solver_eta = solver_eta, solver_accel = solver_accel,
                                    fallback_threshold = fallback_threshold)
      trial_loss = trial_eval$loss
      target = current_loss + armijo_c * step * directional_deriv
      if (directional_deriv >= -1e-12) {
        target = current_loss - 1e-8
      }
      if (trial_loss <= target) {
        alpha = trial_alpha
        weight = trial_weight
        current_loss = trial_loss
        warm_starts = trial_eval$warm_starts
        current_eval = trial_eval
        if (trial_loss < best_loss) {
          best_loss = trial_loss
          best_state = list(alpha = alpha, weight = weight,
                            means = trial_eval$means, predicted = trial_eval$predicted,
                            warm_starts = warm_starts)
        }
        accepted = TRUE
        break
      } else {
        step = step * beta
        backtracks = backtracks + 1
      }
    }
    
    if (!accepted) {
      if (verbose) progressmsg(paste0('Gradient optimizer terminated after iteration ', iter - 1, ' (no improving step)'))
      break
    } else if (verbose) {
      progressmsg(paste0('Gradient iteration ', iter, ' loss: ', sprintf('%.6f', current_loss),
                         ' (backtracks: ', backtracks, ')'))
    }
    
    loss_delta = abs(best_loss - current_loss) / max(1, abs(current_loss))
    if (loss_delta < loss_tol) {
      if (verbose) progressmsg(paste0('Gradient optimizer loss change below threshold (', sprintf('%.3e', loss_delta), ')'))
      break
    }
  }
  
  abs_err = rye.absoluteError(expected = ctx$expected, predicted = best_state$predicted)
  abs_err = cbind(rowMeans(abs_err))
  abs_err = aggregate(abs_err, by = list(ctx$fam[ , 'group']), mean)
  final_error = mean(abs_err[ , 2])
  list(final_error, best_state$alpha, best_state$weight,
       best_state$means, best_state$predicted)
}


rye.optimize = function(X = NULL, fam = NULL,
                        referencePops = NULL, referenceGroups = NULL,
                        alpha = NULL, optimizeAlpha = TRUE,
                        weight = NULL, optimizeWeight = TRUE, attempts = 4,
                        iterations = 100, rounds = 25, threads = 1, startSD = 0.005, endSD = 0.001,
                        populationError = FALSE, early_stop_patience = 0,
                        optimizer = c('gibbs', 'gradient'),
                        grad_max_iter = 50, grad_tol = 1e-5, grad_loss_tol = 1e-6,
                        grad_step = 1, grad_beta = 0.5, grad_armijo_c = 1e-4,
                        grad_fd_eps = 1e-5, grad_simplex = TRUE) {
  
  optimizer = match.arg(optimizer)
  
  if (optimizer == 'gradient') {
    referenceFAM = fam[fam[ , 'population'] %in% referencePops , ]
    referenceX = X[rownames(referenceFAM), ]
    if (is.null(alpha)) {
      alpha = rep(0.001, length(referencePops))
    }
    names(alpha) = referencePops
    if (is.null(weight)) {
      weight = 1 / seq(ncol(X))
    }
    return(rye.optimize.gradient(X = referenceX, fam = referenceFAM,
                                 referencePops = referencePops,
                                 referenceGroups = referenceGroups,
                                 alpha = alpha, weight = weight,
                                 max_iter = grad_max_iter, grad_tol = grad_tol,
                                 loss_tol = grad_loss_tol, step_init = grad_step,
                                 beta = grad_beta, armijo_c = grad_armijo_c,
                                 fd_eps = grad_fd_eps,
                                 project_simplex_weights = grad_simplex))
  }
  
  ## Pull out the reference PCs
  referenceFAM = fam[fam[ , 'population'] %in% referencePops , ]
  referenceX = X[rownames(referenceFAM), ]
  
  ## Start with the shrinking at 0.05 for all pops by default
  if (is.null(alpha)) {
    alpha = rep(0.001, length(referencePops))
  } 
  names(alpha) = referencePops
  
  ## Weights
  if (is.null(weight)) {
    weight = 1 / seq(ncol(X))
  }
  
  allErrors = c()
  
  best_error_so_far = Inf
  no_improve_rounds = 0
  for (round in seq(rounds)) {
    
    sd = startSD - (startSD - endSD) * log(round)/log(rounds)
    if (threads > 1) {
      params = mclapply(seq(attempts), function(i) rye.gibbs(X = referenceX, fam = referenceFAM, referenceGroups = referenceGroups,
                                                            iterations = iterations,
                                                            alpha = alpha, weight = weight, sd = sd,
                                                            optimizeAlpha = optimizeAlpha, optimizeWeight = optimizeWeight), mc.cores = threads)
    } else {
      params = lapply(seq(attempts), function(i) rye.gibbs(X = referenceX, fam = referenceFAM, referenceGroups = referenceGroups,
                                                          iterations = iterations,
                                                          alpha = alpha, weight = weight, sd = sd,
                                                          optimizeAlpha = optimizeAlpha, optimizeWeight = optimizeWeight))
    } 
    
    
    errors = unlist(lapply(params, function(i) i[[1]]))
    
    bestError = which.min(errors)
    meanError = mean(errors)
    progressmsg(paste0('Round ', round, '/', rounds, ' Mean error: ', sprintf("%.6f", meanError),
               ', Best error: ', sprintf('%.6f', errors[bestError])))
    
    bestParams = params[[bestError]]
    alpha = bestParams[[2]]
    weight = bestParams[[3]]
    
    allErrors = c(allErrors, errors[bestError])
    
    if (errors[bestError] < best_error_so_far - 1e-9) {
      best_error_so_far = errors[bestError]
      no_improve_rounds = 0
    } else {
      no_improve_rounds = no_improve_rounds + 1
      if (early_stop_patience > 0 && no_improve_rounds >= early_stop_patience) {
        progressmsg(paste0('Early stopping after ', round, ' rounds; best error: ', sprintf('%.6f', best_error_so_far)))
        break
      }
    }
    
    if (early_stop_patience == 0 && round > 5) {
      errorChange = allErrors[(round - 5):round]
      errorChange = max(errorChange) - min(errorChange)
      if (errorChange <= 0.000025) {
        break
      }
    }
    
  }
  
  return(bestParams)
}

rye = function(eigenvec_file = NULL, eigenval_file = NULL,
               pop2group_file = NULL, output_file = NULL,
               threads = 4, pcs = 20, optim_rounds = 200,
               optim_iter = 100, attempts=4, optim_patience = 0,
               optimizer = 'gibbs', grad_max_iter = 50, grad_tol = 1e-5,
               grad_loss_tol = 1e-6, grad_step = 0.5, grad_beta = 0.5,
               grad_armijo_c = 1e-4, grad_fd_eps = 1e-5, grad_simplex = TRUE){
  ## Perform core operation
  #TODO: Change file reading method to data.table
  logmsg("Reading in Eigenvector file")
  fullPCA = read.table(eigenvec_file, header = FALSE, row.names = NULL)
  rownames(fullPCA) = fullPCA[ , 2]
  logmsg("Reading in Eigenvalue file")
  fullEigenVal = read.table(eigenval_file, header = FALSE, row.names = NULL)[,1]
  logmsg("Reading in pop2group file")
  pop2group = read.table(pop2group_file, header = T, stringsAsFactors = F)
  referenceGroups = pop2group$Group
  names(referenceGroups) = pop2group$Pop
  
  ## Regenerate the FAM from the PCA input
  logmsg("Creating individual mapping")
  fam = as.matrix(fullPCA[ , c(1, 2)])
  colnames(fam) = c('population', 'id')
  rownames(fam) = fam[ , 'id']
  allPops = unique(fam[ , 'population'])
  
  ## Cast PCA to a matrix & scale the PCs
  logmsg("Scaling PCs")
  fullPCA = fullPCA[ , 3:ncol(fullPCA)]
  fullPCA = as.matrix(fullPCA)
  fullPCA = rye.scale(fullPCA)
  
  ## Weight the PCs by their eigenvalues
  logmsg("Weighting PCs")
  weight = fullEigenVal / max(fullEigenVal)
  
  ## Using each region as a population, e.g., combine British and French to WesternEuropean
  logmsg("Aggregating individuals to population groups")
  regionFAM = fam
  regionFAM[fam[,1] %in% names(referenceGroups),1] = referenceGroups[fam[fam[,1] %in% names(referenceGroups), 1]]
  referenceGroups = unique(referenceGroups)
  names(referenceGroups) = referenceGroups
  referencePops = referenceGroups
  
  ## Optimize estimates using NNLS
  logmsg("Optimizing estimates using FISTA-based NNLS solver")
  scaledWeight = weight[seq(pcs)]
  unifAlpha = rep(0.001, length(referencePops))
  names(unifAlpha) = referencePops
  optParams = rye.optimize(X = fullPCA[,seq(pcs)], fam = regionFAM,
                           referencePops = referencePops, referenceGroups = referenceGroups,
                           startSD = 0.01, endSD = 0.005,
                           threads = threads, iterations = optim_iter,
                           rounds = optim_rounds, attempts=attempts,
                           weight = scaledWeight, alpha = unifAlpha, 
                           optimizeWeight = TRUE, optimizeAlpha = TRUE,
                           early_stop_patience = optim_patience,
                           optimizer = optimizer, grad_max_iter = grad_max_iter,
                           grad_tol = grad_tol, grad_loss_tol = grad_loss_tol,
                           grad_step = grad_step, grad_beta = grad_beta,
                           grad_armijo_c = grad_armijo_c, grad_fd_eps = grad_fd_eps,
                           grad_simplex = grad_simplex)
  optWeight = optParams[[3]]
  optMeans = optParams[[4]]
  
  ## Calculate ancestry estimates
  logmsg("Calculate per-individual ancestry estimates")
  optEstimates = rye.predict(X = fullPCA[,seq(pcs)], means = optMeans, weight = optWeight)
  optEstimates = t(apply(optEstimates, 1, function(i) i /sum(i)))
  
  ## Find the mean of each population
  # logmsg("Calculate per-population mean ancestry estimates")
  # optEstimateMeans = do.call(cbind, lapply(allPops, function(i) cbind(apply(t(optEstimates[fam[,1] == i, ,drop = FALSE]), 1, mean))))
  # colnames(optEstimateMeans) = allPops
  optEstimatesAgg = NULL
  for(group in referenceGroups){
    optEstimatesAgg = cbind(optEstimatesAgg, apply(optEstimates[ , group, drop = FALSE], 1, sum))
  }
  colnames(optEstimatesAgg) = as.character(referenceGroups)
  
  ## Create output files
  logmsg("Create output files")
  write.table(x = optEstimatesAgg, 
              file = paste0(output_file, '-', pcs, '.', length(referenceGroups),'.Q'), 
              col.names = TRUE, row.names = TRUE, quote = FALSE, sep = '\t')
  write.table(x = optEstimates, 
              file = paste0(output_file, '-', pcs, '.', ncol(optEstimates), '.Q'),
              col.names = TRUE, row.names = TRUE, quote = FALSE, sep = '\t')
  write.table(x = fam[rownames(optEstimatesAgg), ], 
              file = paste0(output_file, '-', pcs, '.fam'), col.names = TRUE, 
              row.names = TRUE, quote = FALSE, sep = '\t')
  
}


validate_arguments <- function(opt){
  ## Verify the arguments
  #TODO: Move argument validation to its own function
  argumentsGood = TRUE
  if (is.null(opt$eigenvec)) {
    argumentsGood = FALSE
    printError('Eigenvector file not given (--eigenvec)')
  } else if (!file.exists(opt$eigenvec)) {
    argumentsGood = FALSE
    printError(paste('Eigenvector file (--eigenvec=', opt$eigenvec, ') not found'))
  }
  if (is.null(opt$eigenval)) {
    argumentsGood = FALSE
    printError('Eigenvalue file not given (--eigenval)')
  } else if (!file.exists(opt$eigenval)) {
    argumentsGood = FALSE
    printError(paste('Eigenvalue file (--eigenval=', opt$eigenval, ') not found'))
  }
  if (is.null(opt$pop2group)) {
    argumentsGood = FALSE
    printError('Population-to-group mapping file not given (--pop2group)')
  } else if (!file.exists(opt$pop2group)) {
    argumentsGood = FALSE
    printError(paste('Population-to-group mapping file (--pop2group=', opt$pop2group, ') not found'))
  }
  if (is.null(opt$output)) {
    argumentsGood = FALSE
    printError('Output prefix not given (--output)')
  }
  
  #TODO: Implement check for file dimensions
  #TODO: Ensure number of threads don't exceed machine capacity
  
  if (!argumentsGood){
    printError('Incomplete/incorrect arguments were observed, cannot continue.')
    printError(c("Run", script_title, "-h for usage information"))
    # stop("Exiting...") # I don't like the error message
    q(save = "no", status = 1)
  }
}

################################################################################

if (sys.nframe() == 0) {
  optionList = list(
    make_option('--eigenval', type = 'character', default = NULL,
                help = 'Eigenvalue file [REQUIRED]', metavar = '<EVAL_FILE>'),
    make_option('--eigenvec', type = 'character', default = NULL,
                help = 'Eigenvector file [REQUIRED]', metavar = '<EVEC_FILE>'),
    make_option('--pop2group', type = 'character', default = NULL,
                help = 'Population-to-group mapping file [REQUIRED]', metavar = '<P2G_FILE>'),
    make_option('--output', type = 'character', default = "output",
                help = 'Output prefix (Default = output)', metavar = '<OUTPUT_PREFIX>'),
    make_option('--threads', type = 'numeric', default = 4,
                help = 'Number of threads to use (Default = 4)', metavar = '<THREADS>'),
    make_option('--pcs', type = 'numeric', default = 20,
                help = 'Number of PCs to use (Default = 20)', metavar = '<#PCs>'),
    make_option('--rounds', type = 'numeric', default = 200,
                help = 'Number of rounds to use for optimization (higher number = more accurate but slower; Default=200)',
                metavar = '<optim-rounds>'),
    make_option('--iter', type = 'numeric', default = 100,
                help = 'Number of iterations to use for optimization (higher number = more accurate but slower; Default=100)',
                metavar = '<optim-iters>'),
    make_option('--attempts', type = 'numeric', default = 4,
                help = 'Number of attempts to find the optimum values (Default = 4)', metavar = '<ATTEMPTS>'),
    make_option('--stop-patience', type = 'integer', default = 0,
                help = 'Early-stop rounds without improvement (0 = disabled)', metavar = '<ROUNDS>'),
    make_option('--optimizer', type = 'character', default = 'gibbs',
                help = 'Outer optimizer: gibbs (default) or gradient', metavar = '<OPTIMIZER>'),
    make_option('--grad-maxiter', type = 'integer', default = 50,
                help = 'Maximum iterations for gradient optimizer (default = 50)', metavar = '<ITER>'),
    make_option('--grad-tol', type = 'numeric', default = 1e-5,
                help = 'Gradient infinity-norm tolerance (default = 1e-5)', metavar = '<TOL>'),
    make_option('--grad-loss-tol', type = 'numeric', default = 1e-6,
                help = 'Relative loss tolerance for gradient optimizer (default = 1e-6)', metavar = '<TOL>'),
    make_option('--grad-step', type = 'numeric', default = 0.5,
                help = 'Initial step size for gradient optimizer (default = 0.5)', metavar = '<STEP>'),
    make_option('--grad-beta', type = 'numeric', default = 0.5,
                help = 'Backtracking shrink factor beta (default = 0.5)', metavar = '<BETA>'),
    make_option('--grad-eps', type = 'numeric', default = 1e-5,
                help = 'Finite-difference epsilon (default = 1e-5)', metavar = '<EPS>'),
    make_option('--grad-simplex', type = 'logical', default = TRUE,
                help = 'Project gradient updates to the simplex for weights (default = TRUE)', metavar = '<TRUE/FALSE>')
  )
  
  optParser = OptionParser(option_list = optionList)
  opt = parse_args(optParser)
  # Debug only
  # opt = parse_args(optParser, args = c("--eigenvec=extractedChrAllPrunedNoSan.25.eigenvec.gz",
  #                                      "--eigenval=extractedChrAllPrunedNoSan.25.eigenval",
  #                                      "--pop2group=pop2group.txt"))
  # print(opt)
  start_time <- Sys.time()
  logmsg("Parsing user supplied arguments...")
  validate_arguments(opt)
  logmsg("Arguments passed validation")
  logmsg(paste0("Running core rye with ", opt$threads, " threads"))
  rye(eigenvec_file = opt$eigenvec,
      eigenval_file = opt$eigenval,
      pop2group_file = opt$pop2group,
      output_file = opt$output,
      threads = opt$threads,
      attempts = opt$attempts,
      pcs = opt$pcs,
      optim_rounds = opt$rounds,
      optim_iter = opt$iter,
      optim_patience = opt[['stop-patience']],
      optimizer = opt$optimizer,
      grad_max_iter = opt[['grad-maxiter']],
      grad_tol = opt[['grad-tol']],
      grad_loss_tol = opt[['grad-loss-tol']],
      grad_step = opt[['grad-step']],
      grad_beta = opt[['grad-beta']],
      grad_fd_eps = opt[['grad-eps']],
      grad_simplex = isTRUE(opt[['grad-simplex']]))
  logmsg("Process completed")
  end_time <- difftime(Sys.time(), start_time, units = "secs")[[1]]
  #print(end_time)
  logmsg(paste0("The process took ", pretty_time(end_time)))
}
