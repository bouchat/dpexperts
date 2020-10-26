setwd("~/Dropbox/Papers_Submission/PA_EngagingExperts/code")
library(Rcpp)
library(RcppArmadillo)
library(mvtnorm)
library(Bolstad)
library(foreign)
library(ggplot2)
library(DBI)
library(coefplot)
library(RColorBrewer)
library(texreg)
library(reshape2)
library(conquer)
library(MCMCpack)
rm(list = ls())
sourceCpp("eliciteddp-Rcpp.cpp")
source("bayeslm.R")

###############################
######## Simulations #########
###############################
gray.pal <- gray.colors(12, start = 0.2, end = 0.8)

set.seed(200)
n <- 20    ## number of experts
m <- 10    ## number of questions per expert
k <- 10     ## number of covariates
alpha <- 1 ## concentration parameter

# can do this directly:
# cluster <- sample(1:n,n,replace=TRUE,prob=rgamma(n, alpha/n))
## rgamma(n, alpha/n) equivalent to rdirichlet(1, rep(alpha/n, n))
## if we ignore the normalizing constant

cluster <- c(1, rep(2, 11), rep(3, 7), 4)

Variances <- rinvgamma(k*n, shape = 10, scale=.81)	# m*n
Priors <- rnorm(k*n, sd=sqrt(Variances))
sigma<- matrix(Variances, k, n) # inverse gamma m,n
beta <- matrix(Priors,k,n) # normal with variance from inverse gamma
# max.sim.var <- 25
x <- matrix(rnorm(k*m),m,k)
y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + matrix(rnorm(n*m,0,sqrt((x^2) %*% sigma[,cluster])),m,n) #m, n
   # for capped variance:
   # y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + 
         # matrix(rnorm(n*m,0,sqrt((x^2) %*% apply(sigma,1:2,pmin,max.sim.var)[,cluster])),m,n)

samples <- elicited(y, x, discrete=FALSE,
                    max_iter=10000, thin=10, burn_in=1000,
                    prior_lambda = diag(ncol(x)),
                    prior_mu = rep(0, ncol(x)),
                    prior_alpha = 1,
                    prior_beta = 1,
                    concentration = alpha)
same.cluster <- function(x) rep(x, each=length(x)) == rep(x, times=length(x))


###########################
######### Plots ###########
###########################

# n, m, k = 20, 10, 10 
pdf("../figures/clusterexamples-201010.pdf", width = 6, height = 3.5)
par(mfcol=c(1,2))
## Estimate of probability of experts being in same cluster
sam<-round(matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]])),1)
image(1:nrow(samples[[1]]),
      1:nrow(samples[[1]]),
      matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]]))[order(cluster),order(cluster)],
      main = "Estimated probability\nof Same Cluster",
      xlab="Expert", ylab="Expert",
      col=gray.pal)

## Truth of which experts are in same cluster
true<-round(matrix(same.cluster(cluster), length(cluster)),1)
image(1:nrow(samples[[1]]),
      1:nrow(samples[[1]]),
      matrix(same.cluster(cluster), length(cluster))[order(cluster), order(cluster)],
      main = "Truth of Whether in\nSame Cluster",
      xlab="Expert", ylab="Expert",
      col=gray.pal)
sum(abs(true-sam))
dev.off()

# n, m, k = 10, 20, 5

n <- 10    ## number of experts
m <- 20    ## number of questions per expert
k <- 5     ## number of covariates
alpha <- 1 ## concentration parameter

cluster <- c(rep(1, 7), rep(2, 3))

Variances <- rinvgamma(k*n, shape = 10, scale=.81)	#m*n

Priors <- rnorm(k*n, sd=sqrt(Variances))
sigma<- matrix(Variances, k, n) #inverse gamma m,n
beta <- matrix(Priors,k,n) #normal with variance from inverse gamma
#max.sim.var <- 25
x <- matrix(rnorm(k*m),m,k)
y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + matrix(rnorm(n*m,0,sqrt((x^2) %*% sigma[,cluster])),m,n) #m, n
# y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + matrix(rnorm(n*m,0,sqrt((x^2) %*% apply(sigma,1:2,pmin,max.sim.var)[,cluster])),m,n)
samples <- elicited(y, x, discrete=FALSE,
                    max_iter=10000, thin=10, burn_in=1000,
                    prior_lambda = diag(ncol(x)),
                    prior_mu = rep(0, ncol(x)),
                    prior_alpha = 1,
                    prior_beta = 1,
                    concentration = alpha)
same.cluster <- function(x) rep(x, each=length(x)) == rep(x, times=length(x))

# n, m, k = 10, 20, 5
pdf("../figures/clusterexamples-10205.pdf", width = 6, height = 3.5)
par(mfcol=c(1,2))
## Estimate of probability of experts being in same cluster
sam<-round(matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]])),1)
image(1:nrow(samples[[1]]),
      1:nrow(samples[[1]]),
      matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]]))[order(cluster),order(cluster)],
      main = "Estimated probability\nof Same Cluster",
      xlab="Expert", ylab="Expert",
      col=gray.pal)
      
## Truth of which experts are in same cluster
true<-round(matrix(same.cluster(cluster), length(cluster)),1)
image(1:nrow(samples[[1]]),
      1:nrow(samples[[1]]),
      matrix(same.cluster(cluster), length(cluster))[order(cluster), order(cluster)],
      main = "Truth of Whether in\nSame Cluster",
      xlab="Expert", ylab="Expert",
      col=gray.pal)
      sum(abs(true-sam))
      dev.off()

# n, m, k = 10, 10, 10      
n <- 10    ## number of experts
m <- 10    ## number of questions per expert
k <- 10     ## number of covariates
alpha <- 1 ## concentration parameter

cluster <- c(1, rep(2, 3), rep(3, 5), 4)

Variances <- rinvgamma(k*n, shape = 10, scale=.81)	#m*n

Priors <- rnorm(k*n, sd=sqrt(Variances))
sigma <- matrix(Variances, k, n) #inverse gamma m,n
beta <- matrix(Priors,k,n) #normal with variance from inverse gamma
#max.sim.var <- 25
x <- matrix(rnorm(k*m),m,k)
y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + matrix(rnorm(n*m,0,sqrt((x^2) %*% sigma[,cluster])),m,n) #m, n
#y <- x %*% beta[,cluster] + matrix(rnorm(n*m),m,n) + matrix(rnorm(n*m,0,sqrt((x^2) %*% apply(sigma,1:2,pmin,max.sim.var)[,cluster])),m,n)
samples <- elicited(y, x, discrete=FALSE,
                          max_iter=10000, thin=10, burn_in=1000,
                          prior_lambda = diag(ncol(x)),
                          prior_mu = rep(0, ncol(x)),
                          prior_alpha = 1,
                          prior_beta = 1,
                          concentration = alpha)
same.cluster <- function(x) rep(x, each=length(x)) == rep(x, times=length(x))
      
pdf("../figures/clusterexamples-101010.pdf", width = 6, height = 3.5)
par(mfcol=c(1,2))
## Estimate of probability of experts being in same cluster
sam <- round(matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]])),1)
image(1:nrow(samples[[1]]),
            1:nrow(samples[[1]]),
            matrix(apply(apply(samples[[1]], 2, same.cluster), 1, mean), nrow(samples[[1]]))[order(cluster),order(cluster)],
            main = "Estimated probability\nof Same Cluster",
            xlab="Expert", ylab="Expert",
            col=gray.pal)
      
## Truth of which experts are in same cluster
true<-round(matrix(same.cluster(cluster), length(cluster)),1)
image(1:nrow(samples[[1]]),
            1:nrow(samples[[1]]),
            matrix(same.cluster(cluster), length(cluster))[order(cluster), order(cluster)],
            main = "Truth of Whether in\nSame Cluster",
            xlab="Expert", ylab="Expert",
            col=gray.pal)
sum(abs(true-sam))
dev.off()
