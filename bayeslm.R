bayes.lm.mod <- 
function (formula, data, subset, na.action, model = TRUE, x = FALSE, 
    y = FALSE, center = TRUE, prior = NULL, sigma = FALSE) 
{
    ret.x = x
    ret.y = y
    cl = match.call()
    mf = match.call(expand.dots = FALSE)
    m = match(c("formula", "data", "subset", "na.action"), names(mf), 
        0L)
    mf = mf[c(1L, m)]
    mf$drop.unused.levels = TRUE
    mf[[1L]] = quote(stats::model.frame)
    mf = eval(mf, parent.frame())
    mt = attr(mf, "terms")
    y = model.response(mf, "numeric")
    if (is.empty.model(mt)) {
    }
    else {
        x = model.matrix(mt, mf, contrasts)
        if (center) {
            if (is.logical(center)) {
                np = ncol(x)
                x[, 2:np] = scale(x[, 2:np], scale = FALSE, center = TRUE)
            }
        }
        z = z.ls = lm.fit(x, y)
        p1 = 1:z$rank
        z$cov.unscaled = chol2inv(z$qr$qr[p1, p1, drop = FALSE])
        z$prior = prior
        if (!is.null(prior)) {
            if (is.null(prior$P0))
                prior.prec = solve(prior$V0)
            else
                prior.prec = prior$P0
            resVar = if (is.logical(sigma) && !sigma) {
                sum(z$residuals^2)/z$df.residual
            }
            else {
                sigma^2
            }
            ls.prec = solve(resVar * z$cov.unscaled)
            post.prec = prior.prec + ls.prec
            V1 = solve(post.prec)
            b1 = V1 %*% prior.prec %*% prior$b0 + V1 %*% ls.prec %*% 
                coef(z.ls)
            z$post.mean = z$coefficients = as.vector(b1)
            z$post.var = V1
            z$post.sd = sqrt(resVar)
        }
        else {
            resVar = if (is.logical(sigma) && !sigma) {
                sum(z$residuals^2)/z$df.residual
            }
            else {
                sigma^2
            }
            z$post.mean = z$coefficients
            z$post.var = resVar * z$cov.unscaled
            z$post.sd = sqrt(resVar)
        }
        z$fitted.values = x %*% z$post.mean
        z$residuals = y - z$fitted.values
        z$df.residual = nrow(x) - ncol(x)
    }
    class(z) = c("Bolstad", "lm")
    z$na.action = attr(mf, "na.action")
    z$call = cl
    z$terms = mt
    if (model) 
        z$model = mf
    if (ret.x) 
        z$x = x
    if (ret.y) 
        z$y = y
    z
}
