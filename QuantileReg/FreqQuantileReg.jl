module FreqQuantileReg

export quantfreq

using RCall

function quantfreq(y::AbstractVector{T}, X::AbstractMatrix{M},
    Control::Dict{Symbol, <:Any}, sigma::Real = 2, p::Real = 2, tau::Real = 0.5) where {T, M <: Real}
  rcopy(R"""
  Control = $Control
  Y = $y
  X = $X
  Sigma_0 = $sigma
  P_0 = $p
  Tau_0 = $tau

  suppressWarnings(suppressMessages(library(quantreg)))
  suppressWarnings(suppressMessages(library(maxLik)))
  suppressWarnings(suppressMessages(library(AEP)))

  AEPD_est_fun <- function(y,x,beta_0,sigma_0,p_0,tau_0,control){

    #Functions
    ##############################
    logfy_i_fun <- function(y_i,x_i,beta,sigma,p,tau){
      mu <- x_i%*%beta
      if(y_i<=mu){
        delta <- gamma(1+1/p)*abs(mu-y_i)/(sigma*tau)
        logfy_i <- -log(sigma) - delta^p
      }else{
        delta <- gamma(1+1/p)*abs(mu-y_i)/(sigma*(1-tau))
        logfy_i <- -log(sigma) - delta^p
      }
      return(logfy_i)
    }

    loglik_fun <- function(y,x,beta,sigma,p,tau){
      n <- length(y)
      loglik <- 0

      for(i in 1:n){
        y_i <- y[i]
        x_i <- x[i,]
        loglik_i <- logfy_i_fun(y_i,x_i,beta,sigma,p,tau)
        loglik <- loglik + loglik_i
      }
      return(loglik)
    }

    o_fun <- function(par){
      npar <- length(par)
      beta <- par[1:(npar-3)]
      sigma <- par[npar-2]
      p <- par[npar-1]
      tau <- par[npar]
      logfy_i <- logfy_i_fun(y_i,x_i,beta,sigma,p,tau)
      return(logfy_i)
    }

    loss_fun_sigma_p_tau <- function(sigma_p_tau){
      sigma <- abs(sigma_p_tau[1])
      p <- max(c(abs(sigma_p_tau[2]), 1))
      tau <- sigma_p_tau[3]
      n <- length(y)
      loss <- -1.0*loglik_fun(y,x,beta_new,sigma,p,tau)
      return(loss)
    }

    loss_i_fun <- function(y_i,x_i,beta,p,tau){
      mu <- x_i%*%beta

      if(y_i<=mu){
        loss_i <- (abs(mu-y_i)/tau)^p
      }else{
        loss_i <- (abs(mu-y_i)/(1-tau))^p
      }
      return(loss_i)
    }

    #objective function
    loss_fun_beta <- function(beta){
      nbeta <- length(beta)
      n <- length(y)
      loss <- 0

      for(i in 1:n){
        x_i <- x[i,]
        y_i <- y[i]
        loss_i <- loss_i_fun(y_i,x_i,beta,p_curr,tau_curr)
        loss <- loss + loss_i
      }
      return(loss)
    }

    n <- length(y)

    beta_curr <- beta_new <- beta_0
    sigma_curr <- sigma_new <- sigma_0
    p_curr <- p_new <- p_0
    tau_curr <- tau_new <- tau_0



    iter <- 0  #number of Newton-Raphson iterations

    max_abs_change <- 100


    ########################
    ###Optimization
    #######################
    funval_vec <- c()
    while(max_abs_change>control$tol & iter<control$max_iter){

      beta_curr <- beta_new
      sigma_curr <- sigma_new
      p_curr <- p_new
      tau_curr <- tau_new



      par_curr <- c(beta_curr,sigma_curr,p_curr,tau_curr)




      npar <- length(par_curr)

      if(control$est_beta){
        opt_obj_beta <- optim(par=beta_curr,fn=loss_fun_beta,method="Nelder-Mead")
        beta_new <- opt_obj_beta$par
      }else{
        beta_new <- beta_0
      }



      sigma_p_tau_curr <- c(sigma_curr,p_curr,tau_curr)

      opt_obj_sigma_p_tau <- optim(par=sigma_p_tau_curr,fn=loss_fun_sigma_p_tau,method="Nelder-Mead")

      if(control$est_sigma){
        sigma_new <-  abs(opt_obj_sigma_p_tau$par[1])
      }else{
        sigma_new <- sigma_0
      }
      if(control$est_p){
        p_new <- abs(opt_obj_sigma_p_tau$par[2])
      }else{
        p_new <- p_0
      }
      if(control$est_tau){
        tau_new <- abs(opt_obj_sigma_p_tau$par[3])
      }else{
        tau_new <- tau_0
      }

      par_new <- c(beta_new,sigma_new,p_new,tau_new)
      change <- par_new - par_curr


      max_abs_change <- max(abs(par_new - par_curr))

      iter <- iter + 1

      loss_new <- opt_obj_sigma_p_tau$value

      funval_vec[iter] <- loss_new

      if(Control$verbose){
        cat('Iteration number ',iter , ' is finished. The max change is ',max_abs_change,'. The loss is ',loss_new,  '\n', sep='')
      }
    }

    #Calculate gradient and Hessian for standard errors

    if(control$is_se){

      grad_mat <- matrix(NA,nrow=n,ncol=npar)
      hess_mat <- matrix(NA,nrow=n*npar,ncol=npar)
      grad <- matrix(0,nrow=npar,ncol=1)
      hess <- matrix(0,nrow=npar,ncol=npar) #The Hessian

      for(i in 1:n){
        x_i <- x[i,]
        y_i <- y[i]

        grad_mat[i,] <- maxLik::numericGradient(o_fun,t0=par_new)

        grad <- grad + grad_mat[i,]

        hess_mat[((i-1)*npar+1):(i*npar),1:npar] <- maxLik::numericHessian(o_fun,t0=par_new)

        hess <- hess + hess_mat[((i-1)*npar+1):(i*npar),1:npar]

      }
      M <- matrix(0,ncol=npar,nrow=npar)

      for(i in 1:n){
        M <- M + grad_mat[i,]%*%t(grad_mat[i,])
      }

      hess_inv <- solve(hess)

      se <- sqrt(diag(hess_inv%*%M%*%hess_inv))

    }
    #Collect results
    res <- list()
    res$iterations <- iter
    res$funval <- opt_obj_sigma_p_tau$value

    res$funval_vec <- funval_vec
    res$beta <- beta_new
    res$sigma <- sigma_new
    res$p <- p_new
    res$par <- par_new
    res$tau <- tau_new
    res$x <- x
    res$y <- y

    if(control$is_se){
      res$grad_mat <- grad_mat
      res$hess_mat <- hess_mat
      res$grad <- grad
      res$hess <- hess
      res$se <- se
    }
    return(res)
  }

  N <- length(Y)
  #Y <- Y + runif(N)
  #Y <- as.matrix(log(Y))
  if(Control$log){
      Y <- Y + runif(N)
      Y <- as.matrix(log(Y))
    } else {
      Y <- as.matrix(Y)
  }
  NBeta <- dim(X)[2]
  Beta_0 <- rep(0,NBeta)

  AEPD_obj <- AEPD_est_fun(Y,X,Beta_0,Sigma_0,P_0,Tau_0,Control)
  """)
end

end
