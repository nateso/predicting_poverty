library(haven)  
library(gtools)
library(reshape2)
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(magrittr)
library(geosphere)
library(labelled)

rm(list = ls())

#*******************************************************************************
#### merge the different datasets together into one single ####
#*******************************************************************************

eth <- read.csv("../../Data/processed/eth_labels.csv")
mwi <- read.csv("../../Data/processed/mwi_short_labels.csv")
nga <- read.csv("../../Data/processed/nga_labels.csv")
uga_short <- read.csv("../../Data/processed/uga_short_labels.csv")
uga_long <- read.csv("../../Data/processed/uga_long_labels.csv")
tza_short <- read.csv("../../Data/processed/tza_labels_short.csv")
tza_long <- read.csv("../../Data/processed/tza_labels_long.csv")
tza_refresh <- read.csv("../../Data/processed/tza_labels_refresh.csv")








