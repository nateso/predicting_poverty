## analyse attrition in the data

library(dplyr)
library(tidyr)
library(magrittr)
library(stringr)
library(stargazer)

rm(list = ls())

# set the working directory
root_data_dir = '../../Data/lsms/processed/'

#*******************************************************************************
#### Load Data ####
#*******************************************************************************

# load the lsms data without attrition data
lsms_df_hh <- read.csv(paste0(root_data_dir,'labels_hh.csv'))

assets <- c("rooms", 'electric', 'phone', 'radio', 'car', 
            'watsup_qual', 'toilet_qual', 'floor_qual')

#*******************************************************************************
#### No Attrition dataset ####
#*******************************************************************************

# subset the data to the first instance of each household
# and ensure that only housheolds are in the data that are not subject to attrition
no_attr_df <- lsms_df_hh %>% 
  arrange(case_id, wave) %>% 
  distinct(case_id, .keep_all = TRUE) %>% 
  select(country, wave, cluster_id, series, cons_pc_usd_2017, all_of(assets)) %>% 
  mutate(attrition = 0)

#*******************************************************************************
####  Attrition dataset ####
#*******************************************************************************

# generate the attrition dataset
attr_data_files <- list.files(paste(root_data_dir), pattern = 'attr', full.names = T)
mask = unlist(lapply(attr_data_files, function(x){grepl('mwi_long_labels|mwi_short_full',x) == F}))
attr_data_files <- attr_data_files[mask]
attr_df_list <- lapply(attr_data_files, read.csv)
for(i in 1:length(attr_df_list)){
  series = strsplit(attr_data_files[i],'//')[[1]][2]
  series = str_extract_all(series, 'long|short|refresh')[[1]]
  attr_df_list[[i]] %<>% select(country, wave, cluster_id, cons_pc_usd_2017, all_of(assets)) %>%
    mutate(attrition = 1, series = series) 
}
attr_df <- do.call(rbind.data.frame, attr_df_list)


#*******************************************************************************
####  Analyse Attrition ####
#*******************************************************************************
# load helper functions for the PCA.
source("../preprocess_labels/helper_functions.R")

# combine the data
df <- rbind.data.frame(attr_df, no_attr_df)
df$series = paste(df$country, df$series, sep = '_')
df$unique_id = paste('hh_',1:nrow(df))
summary(df$cons_pc_usd_2017)

#...............................................................................
### Create asset index
#...............................................................................

asset_df <- df %>% 
  select(unique_id, all_of(assets))

pca <- create_wealth_index(asset_df)

df$asset_index <- create_wealth_index(asset_df)$asset_index
df$asset_index = (df$asset_index - 0)/sd(df$asset_index, na.rm = T)

#...............................................................................
### Run regressions
#...............................................................................

# run regressions to predict attrition
cons_lm1 <- lm(attrition ~ log(cons_pc_usd_2017) , data = df)
cons_lm2 <- lm(attrition ~ log(cons_pc_usd_2017) + series, data = df) 

asset_lm1 <- lm(attrition ~ asset_index, data = df)
asset_lm2 <- lm(attrition ~ asset_index + series, data = df)

summary(cons_lm1)
summary(cons_lm2)
summary(asset_lm1)
summary(asset_lm2)

# export the results in a latex table

stargazer(cons_lm1, cons_lm2, asset_lm1, asset_lm2,
          covariate.labels = c('log(consumption)', 'Asset Wealth'),
          dep.var.labels = c("P(Attrition = 1|.)"),
          title = "Distortion due to attrition",
          label = "tab:attrition",
          add.lines = c("Series Fixed Effects"),
          type = "latex",
          digits = 2,
          no.space = TRUE,
          align = TRUE,
          omit.stat=c("ser"),
          df = F,
          notes.align = "l",
          notes.label = "",
          omit = c("series"),
          column.sep.width = "-10pt") 

# do t-test to check difference in consumption in both groups
t.test(df$cons_pc_usd_2017[df$attrition == 1], df$cons_pc_usd_2017[df$attrition == 0])

# use the assets as Ind. vars, not the asset index.
asset_lm1 <- lm(paste('attrition ~',paste(assets, collapse = "+")), data = df)
asset_lm2 <- lm(paste('attrition ~',paste(assets, collapse = "+"),'+series'), data = df)

cov_labels = c('log(cons)', 'N rooms', 'Electricity', 'Phone', 'Radio', 'Car', 
               'Motorbike', 'Bike', 'Water Supply', 'Toilet', 'Floor', 'Roof',
               'Wall', 'Cooking Fuel')

stargazer(cons_lm1, cons_lm2, asset_lm1, asset_lm2,
          covariate.labels = cov_labels,
          dep.var.labels = c("P(Attrition = 1|.)"),
          title = "Distortion due to attrition",
          label = "tab:attrition",
          add.lines = c("Series Fixed Effects"),
          type = "latex",
          digits = 2,
          style = "qje",
          no.space = TRUE,
          font.size = "scriptsize",
          align = TRUE,
          omit.stat=c("ser"),
          df = F,
          notes.align = "l",
          notes.label = "",
          omit = c("series"),
          column.sep.width = "-10pt") 

# predict consumptions using attrition as Ind. Var.
lm1 <- lm(log(cons_pc_usd_2017) ~ attrition, data = df)
lm2 <- lm(log(cons_pc_usd_2017) ~ attrition + series, df)
lm3 <- lm(log(cons_pc_usd_2017) ~ attrition + series + series:attrition, df)

 














