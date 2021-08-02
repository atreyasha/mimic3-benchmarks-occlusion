#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(tools)
library(ggplot2)
library(tikzDevice)
library(reshape2)
library(optparse)

plot_abs_bar <- function() {
  # read main csv
  stats <- read.csv("./output/zero/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- abs(stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  mean <- as.data.frame(sapply(4:37, function(i) mean(stats[, i])))
  sd <- as.data.frame(sapply(4:37, function(i) sd(stats[, i])))
  hold <- cbind(mean, sd)
  hold$features <- names(stats)[4:37]
  names(hold)[1:2] <- c("means", "standard_deviations")
  hold$max <- unlist(hold["means"] + hold["standard_deviations"])
  hold$min <- unlist(hold["means"] - hold["standard_deviations"])
  hold$type <- "zero"
  # read main csv
  stats <- read.csv("./output/normal-value/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- abs(stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  mean <- as.data.frame(sapply(4:37, function(i) mean(stats[, i])))
  sd <- as.data.frame(sapply(4:37, function(i) sd(stats[, i])))
  new <- cbind(mean, sd)
  new$features <- names(stats)[4:37]
  names(new)[1:2] <- c("means", "standard_deviations")
  new$max <- unlist(new["means"] + new["standard_deviations"])
  new$min <- unlist(new["means"] - new["standard_deviations"])
  new$type <- "normal-value"
  # combine
  hold <- rbind(hold, new)
  # read main csv
  stats <- read.csv("./output/inner/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- abs(stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  # for each feature, compute mean and standard deviation
  mean <- as.data.frame(sapply(4:37, function(i) mean(stats[, i])))
  sd <- as.data.frame(sapply(4:37, function(i) sd(stats[, i])))
  new <- cbind(mean, sd)
  new$features <- names(stats)[4:37]
  names(new)[1:2] <- c("means", "standard_deviations")
  new$max <- unlist(new["means"] + new["standard_deviations"])
  new$min <- unlist(new["means"] - new["standard_deviations"])
  new$type <- "inner"
  # combine
  hold <- rbind(hold, new)
  # read main csv
  stats <- read.csv("./output/outer/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- abs(stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  # for each feature, compute mean and standard deviation
  mean <- as.data.frame(sapply(4:37, function(i) mean(stats[, i])))
  sd <- as.data.frame(sapply(4:37, function(i) sd(stats[, i])))
  new <- cbind(mean, sd)
  new$features <- names(stats)[4:37]
  names(new)[1:2] <- c("means", "standard_deviations")
  new$max <- unlist(new["means"] + new["standard_deviations"])
  new$min <- unlist(new["means"] - new["standard_deviations"])
  new$type <- "outer"
  # combine
  hold <- rbind(hold, new)
  # make V1 an ordered factor
  hold$features <- factor(hold$features, levels = hold$features[1:34])
  # latex escapes
  levels(hold$features) <- gsub("\\_", "\\\\_", levels(hold$features))
  # make plot
  tikz("abs_bar.tex", width = 22, height = 12, standAlone = TRUE)
  g <- ggplot(data = hold, aes(x = features, y = means, alpha = means)) +
    geom_bar(stat = "identity", fill = "red") +
    geom_errorbar(
      data = hold, aes(ymin = min, ymax = max),
      size = 0.5,
      width = 0.5
    ) +
    xlab("\nOccluded Feature") +
    ylab("Absolute Perturbation\n") +
    theme_bw() +
    theme(
      text = element_text(size = 25),
      axis.text.x = element_text(angle = 90, hjust = 1, size = 10.5),
      plot.title = element_text(hjust = 0.5),
      legend.position = "none"
    ) +
    facet_wrap(~type, ncol = 2, nrow = 2)
  print(g)
  dev.off()
  texi2pdf("abs_bar.tex", clean = TRUE)
  file.remove("abs_bar.tex")
  file.rename("abs_bar.pdf", "./img/abs_bar.pdf")
}

plot_nom_violin <- function() {
  # read main csv
  stats <- read.csv("./output/zero/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- -1 * (stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  hold <- stats[, c(4:37)]
  hold$type <- "zero"
  # read main csv
  stats <- read.csv("./output/normal-value/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- -1 * (stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  new <- stats[, c(4:37)]
  new$type <- "normal-value"
  # combine
  hold <- rbind(hold, new)
  # read main csv
  stats <- read.csv("./output/inner/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- -1 * (stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  new <- stats[, c(4:37)]
  new$type <- "inner"
  # combine
  hold <- rbind(hold, new)
  # next step
  stats <- read.csv("./output/outer/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- -1 * (stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  new <- stats[, c(4:37)]
  new$type <- "outer"
  # combine
  hold <- rbind(hold, new)
  # process for plot
  hold <- melt(hold)
  names(hold)[2] <- "feature"
  levels(hold$feature) <- gsub("\\_", "\\\\_", levels(hold$feature))
  # make plot
  tikz("nom_violin.tex", width = 22, height = 12, standAlone = TRUE)
  g <- ggplot(data = hold, aes(x = feature, y = value)) +
    geom_violin(scale = "width", fill = "red", alpha = 0.8) +
    xlab("\nOccluded Feature") +
    ylab("Nominal Perturbation [Perturbed-Best]\n") +
    theme_bw() +
    theme(
      text = element_text(size = 25),
      axis.text.x = element_text(angle = 90, hjust = 1, size = 10.5),
      plot.title = element_text(hjust = 0.5),
      legend.position = "none"
    ) +
    facet_wrap(~type, nrow = 2, ncol = 2)
  print(g)
  dev.off()
  texi2pdf("nom_violin.tex", clean = TRUE)
  file.remove("nom_violin.tex")
  file.rename("nom_violin.pdf", "./img/nom_violin.pdf")
}

plot_abs_violin <- function() {
  # read main csv
  stats <- read.csv("./output/zero/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- abs(stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  hold <- stats[, c(4:37)]
  hold$type <- "zero"
  # read main csv
  stats <- read.csv("./output/normal-value/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:37)] <- abs(stats[, 3] - stats[, c(4:37)])
  # for each feature, compute mean and standard deviation
  new <- stats[, c(4:37)]
  new$type <- "normal-value"
  # combine
  hold <- rbind(hold, new)
  stats <- read.csv("./output/inner/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- abs(stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  new <- stats[, c(4:37)]
  new$type <- "inner"
  # combine
  hold <- rbind(hold, new)
  # next step
  stats <- read.csv("./output/outer/result.csv", stringsAsFactors = FALSE)
  # find modular difference between true and perturbed
  stats[, c(4:ncol(stats))] <- abs(stats[, 3] - stats[, c(4:ncol(stats))])
  max_number <- max(as.numeric(gsub(".*\\_", "", names(stats[, 4:ncol(stats)])))) + 1
  test <- lapply(1:34, function(i) {
    number <- 4 + ((i - 1) * max_number)
    test <- stats[, c(number:(number + 4))]
    name <- names(test)[1]
    test <- melt(test)[2]
    names(test) <- gsub("\\_\\d+$", "", name)
    return(test)
  })
  test <- do.call(cbind, test)
  stats <- cbind(stats[, c(1:3)], test)
  new <- stats[, c(4:37)]
  new$type <- "outer"
  # combine
  hold <- rbind(hold, new)
  # process for plot
  hold <- melt(hold)
  names(hold)[2] <- "feature"
  levels(hold$feature) <- gsub("\\_", "\\\\_", levels(hold$feature))
  # make plot
  tikz("abs_violin.tex", width = 22, height = 12, standAlone = TRUE)
  g <- ggplot(data = hold, aes(x = feature, y = value)) +
    geom_violin(scale = "width", fill = "red", alpha = 0.8) +
    xlab("\nOccluded Feature") +
    ylab("Absolute Perturbation\n") +
    theme_bw() +
    theme(
      text = element_text(size = 25),
      axis.text.x = element_text(angle = 90, hjust = 1, size = 10.5),
      plot.title = element_text(hjust = 0.5),
      legend.position = "none"
    ) +
    facet_wrap(~type, ncol = 2, nrow = 2)
  print(g)
  dev.off()
  texi2pdf("abs_violin.tex", clean = TRUE)
  file.remove("abs_violin.tex")
  file.rename("abs_violin.pdf", "./img/abs_violin.pdf")
}

check <- list.files("./output/", recursive = TRUE)
if (length(check) == 4) {
  plot_abs_bar()
  plot_abs_violin()
  plot_nom_violin()
} else {
  stop("Wrong number of files")
}
