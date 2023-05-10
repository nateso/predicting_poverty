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
library(kableExtra)

rm(list = ls())

#*******************************************************************************
#### merge the different datasets together into one single ####
#*******************************************************************************

eth_long <- read.csv("../../Data/processed/eth_labels_long.csv") %>% 
  mutate(series = "long")
eth_short <- read.csv("../../Data/processed/eth_labels_short.csv") %>% 
  mutate(series = 'short')
mwi <- read.csv("../../Data/processed/mwi_short_full_labels.csv") %>% 
  mutate(series = 'short')
nga_long <- read.csv("../../Data/processed/nga_long_labels.csv") %>% 
  mutate(series = 'long')
nga_short <- read.csv("../../Data/processed/nga_short_labels.csv") %>% 
  mutate(series = 'short')
uga_short <- read.csv("../../Data/processed/uga_short_labels.csv") %>% 
  mutate(series = 'short')
uga_long <- read.csv("../../Data/processed/uga_long_labels.csv") %>% 
  mutate(series = 'long')
tza_short <- read.csv("../../Data/processed/tza_labels_short.csv") %>% 
  mutate(series = 'short')
tza_long <- read.csv("../../Data/processed/tza_labels_long.csv") %>% 
  mutate(series = 'long')
tza_refresh <- read.csv("../../Data/processed/tza_labels_refresh.csv") %>% 
  mutate(series = 'refresh')

dat <- rbind.data.frame(
  eth_long, eth_short, mwi, nga_long,nga_short, uga_short, uga_long, tza_short,
  tza_long, tza_refresh
)

dat %<>% mutate(unique_id = paste(case_id, wave, sep = '_w')) 

rm(list=setdiff(ls(), "dat"))
summary(dat$cons_pc_usd_2017)

#*******************************************************************************
#### get some descriptive stats by country wave ####
#*******************************************************************************

### get the summary for each country and wave

for(ctry in unique(dat$country)){
  print("*******************************************")
  print(ctry)
  for(wv in unique(dat$wave[dat$country == ctry])){
    print(".........................................")
    print(paste('wave:',wv))
    print('')
    print(summary(dat$cons_pc_usd_2017[dat$country == ctry & dat$wave == wv]))
  }
}

### get simple summary stats for all variables
vnames <- c('rooms', 'electric', 'floor_qual', 'wall_qual', 'roof_qual', 
            'cooking_fuel_qual', 'toilet_qual', 'watsup_qual', 'radio', 
            'tv', 'bike', 'motorcycle', 'fridge', 'car', 'phone', 'hh_size',
            'cons_pc_lcu_2017', 'cons_pc_usd_2017')

for(var in vnames){
  print("........................")
  print(paste('Variable:',var))
  print(summary(dat[[var]]))
  print("........................")
}

# some simple data cleaning
# rooms has three observations with more than 1000 rooms --> assume typo and 
# only keep first two digits
print(sum(dat$rooms> 1000, na.rm= T))
dat %<>% 
  mutate(rooms = ifelse(rooms > 1000,as.numeric(substr(rooms,1,2)), rooms))

# make a graph to check poverty over time by country

plot_dat <- dat %>% 
  select(country, series, start_year, cons_pc_usd_2017) %>% 
  group_by(country, series, start_year) %>% 
  summarise(mean_inc = mean(cons_pc_usd_2017)) %>% ungroup() %>% 
  mutate(ctry_series = paste(country,series,sep = '_'))

poverty_plot <- ggplot(plot_dat) +
  aes(x = start_year, y = mean_inc, color = ctry_series) +
  geom_line() +
  scale_x_continuous(breaks = c(2008:2021)) +
  xlab("Year") + ylab("Average daily per capita consumption in 2017 int. $") +
  theme_bw()

poverty_plot

#*******************************************************************************
#### Do a PCA for the asset index ####
#*******************************************************************************

source("helper_functions.R")

### Repliate the Yeh et al. 2020 index
# extract the variables to replciate the Yeh et al. 2020 asset indice
yeh_assets <- dat %>% select(unique_id, cluster_id, rooms, electric, floor_qual, watsup_qual, 
                             toilet_qual, phone, radio, tv, car)

# impute missing values
yeh_assets <- impute_missing_vals(yeh_assets)

# run the PCA to extract the wealth index
yeh_assets$asset_index <- create_wealth_index(yeh_assets)



### do the same for the more complete asset index (nate index)
nate_assets <- dat %>% 
  select(unique_id, cluster_id, rooms, electric, floor_qual, wall_qual, roof_qual,
         watsup_qual, toilet_qual, phone, radio, tv, car, bike, motorcycle)
nate_assets <- impute_missing_vals(nate_assets)

nate_assets$asset_index <- create_wealth_index(nate_assets)

### store PCA results (wealth index) in the large data
dat %<>% 
  left_join(yeh_assets %>% select(unique_id, asset_index), by = 'unique_id') %>% 
  left_join(nate_assets %>% select(unique_id, asset_index), by = 'unique_id', suffix = c('_yeh','_nate'))


#*******************************************************************************
#### Check correlation of Asset indices and consumption data ####
#*******************************************************************************
cor(dat$asset_index_nate, dat$asset_index_yeh)

cor(dat$asset_index_yeh, dat$cons_pc_usd_2017)
cor(dat$asset_index_yeh, log(dat$cons_pc_usd_2017))

cor(dat$asset_index_nate, dat$cons_pc_usd_2017)
cor(dat$asset_index_nate, log(dat$cons_pc_usd_2017))

#*******************************************************************************
#### Aggregate data at cluster level ####
#*******************************************************************************

# check out the number of households per cluster and wave (should be consistent)
# remove those clusters with less than 3? hosueholds (??) --> estimate will be too noisy.

cl_dat <- dat %>% 
  select(country, start_month, start_year, end_month, end_year, wave, series, cluster_id,
         rural, lat, lon, cons_pc_usd_2017, cons_pc_lcu_2017, asset_index_yeh, asset_index_nate) %>% 
  group_by(country, start_month, start_year, end_month, end_year, wave, series, cluster_id,
           rural, lat, lon) %>% 
  summarise(mean_pc_cons_usd_2017 = mean(cons_pc_usd_2017),
            median_pc_cons_usd_2017 = median(cons_pc_usd_2017),
            mean_pc_cons_lcu_2017 = mean(cons_pc_lcu_2017),
            median_pc_cons_lcu_2017 = median(cons_pc_lcu_2017),
            mean_asset_index_nate = mean(asset_index_nate),
            median_asset_index_nate = median(asset_index_nate),
            mean_asset_index_yeh = mean(asset_index_yeh),
            median_asset_index_yeh = median(asset_index_yeh),
            n_households = n()) %>% 
  mutate(extreme_poor = ifelse(median_pc_cons_usd_2017 < 2.15,1,0)) %>% 
  ungroup()

#*******************************************************************************
#### Add a start and end time stamp and unique id####
#*******************************************************************************

cl_dat %<>% 
  mutate(start_day = 1, end_day = 30) %>% 
  mutate(start_ts = as.Date(paste(start_year, start_month, start_day, sep = "-"))) %>% 
  mutate(end_ts = start_ts + 365) %>% 
  mutate(unique_id = paste(cluster_id,start_year,sep = '_')) %>%
  relocate(country, start_day, start_month, start_year, end_day, end_month, end_year, 
           start_ts, end_ts, wave, series, cluster_id, rural, unique_id)

#*******************************************************************************
#### Descriptive stats on the cluster dataset ####
#*******************************************************************************

vnames <- names(cl_dat)[grepl("median|mean|n_|extreme_poor",names(cl_dat))]

for(var in vnames){
  print("........................")
  print(paste('Variable:',var))
  print(summary(cl_dat[[var]]))
  print("........................")
}


cor(cl_dat$mean_pc_cons_usd_2017, cl_dat$mean_asset_index_nate)
cor(cl_dat$mean_pc_cons_usd_2017, cl_dat$mean_asset_index_yeh)

cor(log(cl_dat$mean_pc_cons_usd_2017), cl_dat$mean_asset_index_nate)
cor(log(cl_dat$mean_pc_cons_usd_2017), cl_dat$mean_asset_index_yeh)

cor(cl_dat$median_pc_cons_usd_2017, cl_dat$mean_asset_index_nate)
cor(cl_dat$median_pc_cons_usd_2017, cl_dat$mean_asset_index_yeh)

cor(log(cl_dat$median_pc_cons_usd_2017), cl_dat$mean_asset_index_nate)
cor(log(cl_dat$median_pc_cons_usd_2017), cl_dat$mean_asset_index_yeh)



#*******************************************************************************
#### Final Sample table ####
#*******************************************************************************

cl_dat$country_series <- paste(cl_dat$country, cl_dat$series, sep = "_") 
tab <- table(cl_dat$start_year, cl_dat$country_series)
tab
#kable(tab,format = 'latex')


#*******************************************************************************
#### Save the final dataset ####
#*******************************************************************************

write.csv(dat, "../../Data/processed/labels_hh.csv", row.names = F)
write.csv(cl_dat,"../../Data/processed/labels_cluster.csv",row.names = F)

















