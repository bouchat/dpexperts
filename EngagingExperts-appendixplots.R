library(fields)

expertgrid.1 <- cbind(
 c(1, 1, 1, 1),
 c(1, 1, 1, 1),
 c(1, 1, 1, 1),
 c(1, 1, 1, 1)
)

expertgrid.2 <- cbind(
 c(1, 1, 0, 0),
 c(1, 1, 0, 0),
 c(0, 0, 1, 1),
 c(0, 0, 1, 1)
)

expertgrid.3 <- cbind(
 c(1, 1, 0, 0),
 c(1, 1, 0, 0),
 c(0, 0, 1, 0),
 c(0, 0, 0, 1)
)

expertgrid.3u <- cbind(
 c(1, 1, 0.3, 0),
 c(1, 1, 0.3, 0),
 c(0.3, 0.3, 1, 0),
 c(0, 0, 0, 1)
)

expertgrid.4 <- cbind(
 c(1, 0, 0, 0),
 c(0, 1, 0, 0),
 c(0, 0, 1, 0),
 c(0, 0, 0, 1)
)

expertgrid.2u <- cbind(
 c(1, .5, 0, 0),
 c(.5, 1, 0, 0),
 c(0, 0, 1, .8),
 c(0, 0, .8, 1)
)


expert.grids<-list(expertgrid.1, expertgrid.2, expertgrid.3, expertgrid.4, expertgrid.3u, expertgrid.2u)

gray.pal <- gray.colors(12, start = 0.2, end = 0.8)

pdf("../figures/appendixplots.pdf", width = 7, height = 9)
par(mfrow=c(3,2))
label <- letters[1:length(expert.grids)]
for(i in 1:length(expert.grids)){
		image.plot(1:nrow(expert.grids[[i]]),
    1:nrow(expert.grids[[i]]),
    expert.grids[[i]],
    main = paste0("Estimated Probability of Same Cluster (",label[i], ")"),
		xlab="Expert", ylab="Expert", zlim=c(0,1),
		col=gray.pal)
}
dev.off()