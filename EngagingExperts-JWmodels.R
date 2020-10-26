setwd("~/Dropbox/Papers_Submission/PA_EngagingExperts/code")
library(Rcpp)
library(RcppArmadillo)
library(mvtnorm)
library(Bolstad)
library(foreign)
library(ggplot2)
library(reshape2)
library(conquer)
library(MCMCpack)
library(fields)
sourceCpp("eliciteddp-Rcpp.cpp")

################################
####### Data Preparation #######
################################

# Load Western/Jackman data
western94 <- read.table("western94.asc", header = T, sep = " ")
attach(western94)

# Prior means on covariates for hypothetical experts
Wallerstein <- c(0, .3, -5, 0)
Stephens <- c(0, .3, 0, 10)
Econ.Skeptic <- c(0, .3, 0, -5)
Communist.1 <- c(0, .9, 0, 0)
Communist.2 <- c(0, .1, 5, 0)
Neoliberal <- c(0, -.1, 0, -10)
Uncertain.1 <- c(0, .1, .1, .1)
Uncertain.2 <- c(0, -.1, -.1, -.1)
Labor.1 <- c(0, .1, 5, 0)
Labor.2 <- c(0, .1, 10, 0)

# Matrix of expert prior means with expert and covariate name labels
JackmanWestern <- c(Wallerstein,Stephens)
Priors <- data.frame(Wallerstein, Stephens, Econ.Skeptic, Communist.1, Communist.2, Neoliberal, Uncertain.1, Uncertain.2, Labor.1, Labor.2)
Expert <- colnames(Priors)
Intercept <- Priors[1,]
LeftGovt <- Priors[2,]
LogLabor <- Priors[3,]
EconConcen <- Priors[4,]
Priors$variable <- c("Intercept", "Left Government", "Log Labor Force", "Economic Concentration")
Priors.long <- melt(Priors, id.vars = "variable", variable.name = "Expert",  value.name = "mean")

# Remove variable names for matrix
Priors.mat <- Priors[,-11]

# Simple average prior across experts
avg <- apply(Priors.mat, 1, mean)


# Prior variances on covariates for hypothetical experts
Wallerstein.var <- c(100000, .15^2, 2.5^2, 10^12)
Stephens.var <- c(100000, .15^2, 10^12, 5^2)
Econ.Skeptic.var <- c(100000, .1^2, 0, 2.5^2)
Communist.1.var <- c(100000, .5^2, 10^4, 10^4)
Communist.2.var <- c(100000, .05^2, 2.5^2, 10^4)
Neoliberal.var <- c(100000, .05^2, 0, 3^2)
Uncertain.1.var <- c(100000, 10^4, 10^4, 10^4)
Uncertain.2.var <- c(100000, 10^4, 10^4, 10^4)
Labor.1.var <- c(100000, .05^2, 2.5^2, 10^4)
Labor.2.var <- c(100000, .05^2, 3^2, 10^4)

JackmanWestern.var <- c(Wallerstein.var,Stephens.var)
Variances <- data.frame(Wallerstein.var, Stephens.var, Econ.Skeptic.var, 
                        Communist.1.var, Communist.2.var, Neoliberal.var, 
                        Uncertain.1.var, Uncertain.2.var, Labor.1.var, 
                        Labor.2.var)

# Average of prior variances
avg.var <- apply(Variances, 1, mean)

# Prior precisions on covariates for hypothetical experts (for plotting)
Wallerstein.prec <- c(0, (1/0.0225), (1/6.25), 0)
Stephens.prec <- c(0, (1/0.0225), 0, (1/25))
Econ.Skeptic.prec <- c(0, (1/.01), 0, (1/6.25))
Communist.1.prec <- c(0, (1/.25), (1/10^4), (1/10^4))
Communist.2.prec <- c(0, (1/.0025), (1/2.5^2), (1/10^4))
Neoliberal.prec <- c(0, (1/.0025), 0, (1/9))
Uncertain.1.prec <- c(0, (1/10^4), (1/10^4), (1/10^4))
Uncertain.2.prec <- c(0, (1/10^4), (1/10^4), (1/10^4))
Labor.1.prec <- c(0, (1/.0025), (1/6.25), (1/10^4))
Labor.2.prec <- c(0, (1/.0025), (1/9), (1/10^4))

# Expert prior precisions with expert and covariate name labels
JackmanWestern.prec <- c(Wallerstein.prec, Stephens.prec)
Precision <- data.frame(Wallerstein.prec, Stephens.prec, Econ.Skeptic.prec, 
                        Communist.1.prec, Communist.2.prec, Neoliberal.prec, 
                        Uncertain.1.prec, Uncertain.2.prec, Labor.1.prec, 
                        Labor.2.prec)
Precision$variable <- c("Intercept", "Left Government", "Log Labor Force", 
                      "Economic Concentration")
Precision.long <- melt(Precision, id.vars = "variable", variable.name = "Expert",  
                     value.name = "precision")
Data <- Priors.long
Data$precision <- Precision.long$precision

# Average of expert prior precisions with removed variable names from matrix
Precision.mat <- Precision[,-11]
avg.prec <- apply(Precision.mat, 1, mean)

#####################################
####### Model with Avg Priors #######
#####################################

source("bayeslm.R")
results.avg <- bayes.lm.mod(union ~ left + size + concen, data = western94, 
                            model = TRUE, x = FALSE, y = FALSE, center = TRUE, 
                            prior = list(b0 = avg, P0 = diag(avg.prec,4)), 
                            sigma = FALSE)


###################################
####### DP Model Clustering #######
###################################

# Get clusters for DP model
set.seed(3000)

n <- 10    ## number of experts
m <- 4    ## number of questions per expert
k <- 4     ## number of covariates
alpha <- 1 ## concentration parameter


cluster <- rdp_cluster(n, alpha)
    # can do this directly:
    # cluster <- sample(1:n,n,replace=TRUE,prob=rgamma(n, alpha/n))
    ## rgamma(n, alpha/n) equivalent to rdirichlet(1, rep(alpha/n, n))
    ## if we ignore the normalizing constant

sigma <- as.matrix(Variances, m, n)
beta <- as.matrix(Priors.mat, k, n)
max.sim.var <- 25 # set cap on variance in simulations to keep sensible values
x <- matrix(rnorm(k*m), m, k)
y <- x %*% beta[,cluster] + 
      matrix(rnorm(n*m), m, n) + 
      matrix(rnorm(n*m, 0, sqrt((x^2) %*% apply(sigma,1:2, pmin, max.sim.var)[,cluster])), m, n)

samples <- elicited(y, x, discrete = FALSE,
                    max_iter = 10000, thin = 10, burn_in = 1000,
                    prior_lambda = diag(ncol(x)),
                    prior_mu = rep(0, ncol(x)),
                    prior_alpha = 1,
                    prior_beta = 1,
                    concentration = alpha)

# Cluster assignment of experts from samples
same.cluster <- function(x) rep(x, each = length(x)) == rep(x, times = length(x))
clust.assn <- round(matrix(apply(apply(samples$assignment, 2, same.cluster), 1, mean), 
                           nrow(samples$assignment)),1)

# Assigning equal weight to every cluster to compare to averaging
weights <- rep(1/nrow(unique(clust.assn)), nrow(unique(clust.assn)))

# Clustered model parameters
clust.beta <- weights %*% unique(clust.assn) %*% t(beta)
# clust.prec <- weights %*% unique(clust.assn) %*% t(1/sigma)
clust.var <- weights %*% unique(clust.assn) %*% t(sigma)

# Transform to numeric
clust.beta <- as.numeric(clust.beta)
# clust.prec <- as.numeric(clust.prec)
clust.var <- as.numeric(clust.var)

# Use capped simulated variance estimates with cluster assignments
clust.varcap <- weights %*% unique(clust.assn) %*% t(apply(sigma,1:2,pmin,max.sim.var)[,cluster])
clust.varcap <- as.numeric(clust.varcap)

# Create vectors of clustered parameters
clust.var.vec <- as.vector(clust.var)
# clust.prec.vec <- as.vector(clust.prec)
clust.varcap.vec <- as.vector(clust.varcap)


###################################
###### Plot Clustered Experts #####
###################################

expert.order <- apply(clust.assn, 2, FUN = function(x) which(x > 0.5))
expert.order <- unlist(unique(expert.order))

expert.names <- c("Wallerstein", "Stephens", "Econ Skeptic", "Communist 1", 
                "Communist 2", "Neoliberal", "Uncertain 1", "Uncertain 2", 
                "Labor 1", "Labor 2")

# Plot expert clusters in matrix
gray.pal <- gray.colors(12, start = 0.2, end = 0.8)

pdf("../figures/jw_clustered_experts.pdf", width = 6, height = 6)
par(mar = c(6, 6, 6, 6))
image(clust.assn[expert.order, expert.order], 
      col = gray.pal, axes = F)
mtext(text = expert.names[expert.order], 
      side = 2, line = 0.3, 
      at = seq(0, 1, length = length(expert.names)), las = 1)
mtext(text = expert.names[expert.order], 
      side = 1, line = 0.3, 
      at = seq(0, 1, length = length(expert.names)), las = 2)
image.plot(clust.assn, legend.only = T, col = gray.pal)
dev.off()


#################################
####### Model Comparisons #######
#################################

source("bayeslm.R")

# Using clustered priors with capped variances; 
# noncapped variances are also feasible if elicited prior variances are more 
# informative or with researcher hyperprior imposed
results.clustvarcap <- bayes.lm(union ~ left + size + concen, data = western94, 
                              model = TRUE, x = FALSE, y = FALSE, center = TRUE, 
                              prior = list(b0 = clust.beta, V0 = diag(clust.varcap.vec, 4), 
                                           sigma = FALSE))


# Comparing results from averaging to results from DP clustering
se.avg <- summary(results.avg)$std.err
se.clust <- summary(results.clustvarcap)$std.err
se.compare <- data.frame(se.avg, se.clust)

post.mean.avg <- results.avg$post.mean
post.mean.clust <- results.clustvarcap$post.mean
post.mean.compare <- data.frame(post.mean.avg, post.mean.clust)

cred.int.95 <- data.frame(ci.95 = rep(NA, 4))
cred.int.5 <- data.frame(ci.5 = rep(NA, 4))
for(i in 1:4){
  for(j in 1:2){
    cred.int.95[i,j] <- post.mean.compare[i,j]+2.09*(se.compare[i,j]/sqrt(19))
    cred.int.5[i,j] <- post.mean.compare[i,j]-2.09*(se.compare[i,j]/sqrt(19))
  }}


models <- data.frame(model.name = c((rep("Average", 4)), (rep("DP", 4))))
models$variable <- (rep(c("Intercept", "Left Govt", "Log Labor", "Econ Concentration"), 2))
models$variable.label <- factor(models$variable,
                           levels = c("Intercept", "Left Govt", "Log Labor", "Econ Concentration"))

models$post.mean <- c(post.mean.avg, post.mean.clust)
models$se <- c(se.avg, se.clust)
models$ci.95 <- c(cred.int.95$ci.95, cred.int.95$V2)
models$ci.5 <- c(cred.int.5$ci.5, cred.int.5$V2)

###################
#### Plotting ####
###################

#######################
#### original priors ##
#######################

Data$y <- seq(1:10)
Data$y.position <- -1*as.numeric(as.factor(Data$Expert))
Data$Expert <- gsub("[.]", " ", Data$Expert)
Data <- Data[Data$variable != "Intercept", ]
Data$prec.mag <- ifelse(Data$precision>10,"high","low")

# grayscale with errorbars
ggplot(Data, aes(x=mean, y=y.position)) +
  geom_vline(xintercept=0, lwd=0.5, colour="grey50") +
  geom_errorbarh(aes(xmin=(mean - .5*(precision)), xmax=(mean+.5*(precision))),
                 lwd=1.5, height = 0.5, colour="gray60") +
  theme(text = element_text(size=16))+
  geom_point(size=4, pch=21, fill="gray40") +
  theme_bw() + facet_grid(. ~ variable, scales="free_x") + 	
  labs(x="Prior Mean and Precision", y="Expert") + 	
  scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +
  scale_y_continuous(breaks = unique(Data$y.position),
                     labels = unique(Data$Expert))
ggsave("../figures/jw_priors_errorbars.pdf", width = 6, height = 6)

# grayscale with high/low precision colors
Data$Precision<-ifelse(Data$precision>10,"High (>10)","Low (<10)")
ggplot(Data, aes(x=mean, y=y.position, fill=Precision)) +
  geom_vline(xintercept=0, lty=2, lwd=1, colour="grey70") +
  geom_point(size=4, pch=21) +
  theme_bw() + facet_grid(. ~ variable, scales="free_x") + 
  labs(x="Prior Mean", y="Expert") + 
  scale_y_continuous(breaks = unique(Data$y.position), labels = unique(Data$Expert)) + 
  scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +
  scale_fill_manual(values=c("gray80", "grey40")) +
  theme(legend.position = "bottom")
ggsave("../figures/jw_priors_color.pdf", width = 6, height = 4)


################################
#### Model results comparison #
################################

# grayscale
ggplot(models, aes(x=post.mean, y=model.name, fill=model.name)) +
  geom_vline(xintercept=0, lty=2, lwd=1, colour="grey70") +
  geom_errorbarh(aes(xmin=ci.5, xmax=ci.95, height=.15),
                 lwd=1.5, colour="gray60") +
  geom_point(size=4, pch=21, fill="gray40") +
  theme_bw()  + labs(x="Posterior Mean and 95% Credible Interval", y="Model") + 
  scale_fill_manual(values=c("turquoise2", "red1")) + 
  scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +
  facet_grid(. ~ variable, scales="free_x")
ggsave("../figures/jw_model_results_comparison.pdf", width = 6, height = 4)


################################################
#### Jackman/Western original means/precisions
################################################

variables <- factor(models$variable,
                  levels = c("Intercept", "Left Govt", "Log Labor", "Econ Concentration"))

expert <- c(rep("Wallerstein",4), (rep("Stephens",4)))

JackmanWestern.all <- data.frame(JackmanWestern, JackmanWestern.prec, variables, expert)

# grayscale
ggplot(JackmanWestern.all, aes(x=JackmanWestern, y=expert)) +
  geom_vline(xintercept=0, lty=2, lwd=1, colour="grey50") +
  geom_errorbarh(aes(xmin=(JackmanWestern - .5*(JackmanWestern.prec)), 
                     xmax=(JackmanWestern+.5*(JackmanWestern.prec))),
                 lwd=1.5, colour="gray60", height=.15) +
  geom_point(size=4, pch=21, fill="gray40") +
  scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +
  theme_bw() + facet_grid(. ~ variables, scales="free_x") + 
  theme(panel.spacing = unit(12, "pt")) +
  labs(x="Prior Mean and Precision", y="Expert")
ggsave("../figures/jw_original.pdf", width = 6, height = 4)

