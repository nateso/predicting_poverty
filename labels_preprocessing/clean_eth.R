# Some of this code is inspired by the sustainBench package
# online available at: https://github.com/sustainlab-group/sustainbench

library(haven)
library(gtools)
library(reshape2)
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(magrittr)
library(geosphere)

rm(list = ls())

#*******************************************************************************
#### ETHIOPIA ####
#*******************************************************************************

et1_house = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/sect9_hh_w1.dta") 
et1_ass = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/sect10_hh_w1.dta")
et1_cons = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/cons_agg_w1.dta")

et2_house = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2013/sect9_hh_w2.dta")
et2_ass = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2013/sect10_hh_w2.dta")
et2_cons = read_dta('../../Data/lsms/Ethiopia/first_ESS/ETH_2013/cons_agg_w2.dta')

et3_a <- read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Household/sect_cover_hh_w3.dta")
et3_house = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Household/sect9_hh_w3.dta")
et3_ass = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Household/sect10_hh_w3.dta")
et3_cons = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/cons_agg_w3.dta")

et1_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/pub_eth_householdgeovariables_y1.dta")
et2_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2013/Pub_ETH_HouseholdGeovars_Y2.dta")
et3_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Geovariables/ETH_HouseholdGeovars_y3.dta")

#*******************************************************************************
##### ID variables #####
#*******************************************************************************
# In Ethiopia, the sample was extended to urban households in wave 2. Thus,
# the main id variable comes from wave 2

# use the geovariables to identify whether a household moved outside the EA
et1_geos %<>% select(household_id, ea_id, LAT_DD_MOD, LON_DD_MOD) %>% 
  rename(lat_1 = LAT_DD_MOD, lon_1 = LON_DD_MOD)

et2_geos %<>% select(household_id, household_id2, ea_id, ea_id2, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_2 = lat_dd_mod, lon_2 = lon_dd_mod)

et3_geos %<>% select(household_id2, ea_id2, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_3 = lat_dd_mod, lon_3 = lon_dd_mod) %>% 
  left_join(et3_house %>% select(household_id, household_id2, ea_id), by = 'household_id2')

# the EA variables
long_panel_eas <- et1_house %>% 
  select(ea_id) %>% distinct() %>% 
  left_join(et2_house %>% filter(household_id != "") %>% select(ea_id, ea_id2) %>% distinct(), by = 'ea_id') %>% 
  left_join(et1_geos %>% select(ea_id, lat_1, lon_1) %>% distinct(), by = 'ea_id')

short_panel_eas <- et2_house %>%
  select(ea_id2) %>% distinct() %>% filter((ea_id2 %in% long_panel_eas$ea_id2) == F) %>% 
  left_join(et2_geos %>% select(ea_id2, lat_2, lon_2) %>% filter(is.na(lat_2) == F) %>% distinct(), by = 'ea_id2')


# long panel # this removes all split-off households
w1_ids <- et1_house %>% 
  select(household_id, ea_id) %>% 
  left_join(et1_cons %>% select(household_id, total_cons_ann), by = 'household_id') %>% 
  filter(is.na(total_cons_ann) == F) %>% # ensure that consumption data is available
  select(-total_cons_ann) 

w2_ids <- et2_house %>% 
  select(household_id, household_id2, ea_id, ea_id2) %>% 
  filter(household_id %in% w1_ids$household_id) %>% 
  left_join(long_panel_eas %>% select(ea_id, lat_1, lon_1), by = 'ea_id') %>% 
  left_join(et2_geos %>% select(household_id2, lat_2, lon_2), by = 'household_id2') %>% 
  mutate(dist_y1y2 = distHaversine(cbind(lon_2,lat_2),cbind(lon_1,lat_1))) %>% 
  mutate(dist_y1y2 = ifelse(is.na(dist_y1y2),0,dist_y1y2)) %>% # assume that if no GPS data, household did not move (only one case)
  filter(dist_y1y2 < 1) %>% # remove households that outside of EA
  select(-dist_y1y2, -lat_2, -lon_2) %>% 
  left_join(et2_cons %>% select(household_id2, total_cons_ann), by = 'household_id2') %>% 
  filter(is.na(total_cons_ann) == F) %>% 
  select(-total_cons_ann)
  
w3_ids <- et3_house %>% 
  select(household_id, household_id2, ea_id, ea_id2) %>% 
  filter(household_id2 %in% w2_ids$household_id2) %>% 
  left_join(long_panel_eas %>% select(ea_id, lat_1, lon_1), by = 'ea_id') %>% 
  left_join(et3_geos %>% select(household_id2, lat_3, lon_3), by = 'household_id2') %>% 
  mutate(dist_y1y3 = distHaversine(cbind(lon_3,lat_3),cbind(lon_1,lat_1))) %>% 
  filter(dist_y1y3 < 1) %>% # remove households that outside of EA
  select(-dist_y1y3, -lat_3, -lon_3) %>% 
  left_join(et3_cons %>% select(household_id2, total_cons_ann), by = 'household_id2') %>% 
  filter(is.na(total_cons_ann) == F) %>% 
  select(-total_cons_ann)
  

# urban household (aka short panel only starting wave 2)
# this removes split-off households and households that dropped out of the sample
uw2_ids <- et2_house %>%
  filter(household_id == '') %>% 
  select(household_id2, ea_id2) %>% 
  left_join(et2_cons %>% select(household_id2, total_cons_ann), by = 'household_id2') %>% 
  filter(is.na(total_cons_ann) == F) %>% 
  select(-total_cons_ann)

uw3_ids <- et3_house %>%
  filter(household_id2 %in% uw2_ids$household_id2) %>% 
  select(household_id2, ea_id2) %>% 
  left_join(short_panel_eas %>% select(ea_id2, lat_2, lon_2), by = 'ea_id2') %>% 
  left_join(et3_geos %>% select(household_id2, lat_3, lon_3), by = 'household_id2') %>% 
  mutate(dist_y2y3 = distHaversine(cbind(lon_3,lat_3),cbind(lon_2,lat_2))) %>% 
  filter(dist_y2y3 < 1) %>% # remove households that outside of EA
  select(-dist_y2y3, -lat_3, -lon_3) %>% 
  left_join(et3_cons %>% select(household_id2, total_cons_ann), by = 'household_id2') %>% 
  filter(is.na(total_cons_ann) == F) %>% 
  select(-total_cons_ann)
  
# short_panel_ids
short_panel_ids <- uw3_ids %>%
  rename(lat = lat_2, lon = lon_2)

long_panel_ids <- w3_ids %>% 
  rename(lat = lat_1, lon = lon_1)


# In Ethiopia, the households get new lats and lons if they moved more than 10 km
# I use this information to exclude households that moved outside the ea. (i.e. if they have new coordinates)
#*******************************************************************************
##### Housing characteristics #####
#*******************************************************************************

et1_house = et1_house %>%
  select(household_id, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0))

et2_house = et2_house %>%
  select(household_id2, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0)) 

et3_house = et3_house %>%
  select(household_id2, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q13_e, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0),
         watsup = ifelse(is.na(watsup), hh_s9q13_e, watsup)) %>% 
  select(-hh_s9q13_e)

#*******************************************************************************
###### Recode Housing ######
#*******************************************************************************
floor = read.csv("../../Data/lsms/Ethiopia/recode/floor_recode.csv")
wall = read.csv("../../Data/lsms/Ethiopia/recode/wall_recode.csv")
roof = read.csv("../../Data/lsms/Ethiopia/recode/roof_recode.csv")
cooking_fuel = read.csv("../../Data/lsms/Ethiopia/recode/cooking_fuel_recode.csv")
toilet = read.csv("../../Data/lsms/Ethiopia/recode/toilet_recode.csv")
watsup = read.csv("../../Data/lsms/Ethiopia/recode/watsup_recode.csv")

et1_house %<>% 
  left_join(floor,by = c('floor' = 'floor_code')) %>% 
  left_join(wall,by = c('wall' = 'wall_code')) %>% 
  left_join(roof,by = c('roof' = 'roof_code')) %>% 
  left_join(cooking_fuel,by = c('cooking_fuel' = 'cooking_fuel_code')) %>% 
  left_join(toilet,by = c('toilet' = 'toilet_code')) %>% 
  left_join(watsup,by = c('watsup' = 'watsup_code')) %>% 
  select(-floor, -wall, -roof, -cooking_fuel, -toilet, -watsup)

et2_house %<>% 
  left_join(floor,by = c('floor' = 'floor_code')) %>% 
  left_join(wall,by = c('wall' = 'wall_code')) %>% 
  left_join(roof,by = c('roof' = 'roof_code')) %>% 
  left_join(cooking_fuel,by = c('cooking_fuel' = 'cooking_fuel_code')) %>% 
  left_join(toilet,by = c('toilet' = 'toilet_code')) %>% 
  left_join(watsup,by = c('watsup' = 'watsup_code')) %>% 
  select(-floor, -wall, -roof, -cooking_fuel, -toilet, -watsup)

# the variables for wave 3 change slightly. Thus adapt recode...
watsup = read.csv("../../Data/lsms/Ethiopia/recode/watsup_recode_w3.csv")
toilet = read.csv("../../Data/lsms/Ethiopia/recode/toilet_recode_w3.csv")

et3_house %<>% 
  left_join(floor,by = c('floor' = 'floor_code')) %>% 
  left_join(wall,by = c('wall' = 'wall_code')) %>% 
  left_join(roof,by = c('roof' = 'roof_code')) %>% 
  left_join(cooking_fuel,by = c('cooking_fuel' = 'cooking_fuel_code')) %>% 
  left_join(toilet,by = c('toilet' = 'toilet_code')) %>% 
  left_join(watsup,by = 'watsup') %>% 
  select(-floor, -wall, -roof, -cooking_fuel, -toilet, -watsup)

rm(floor, wall, roof, cooking_fuel, toilet, watsup)
#*******************************************************************************
##### Assets #####
#*******************************************************************************

et1_ass %<>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(household_id, hh_s10q0a, hh_s10q01) %>% 
  mutate(hh_s10q01 = ifelse(hh_s10q01 > 0,1,0)) %>% 
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>% 
  mutate(phone = ifelse(`Fixed line telephone` + `Mobile Telephone` > 0,1,0)) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  rename(radio=Radio, tv=Television, bike = Bicycle,
          motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

et2_ass %<>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio/tape recorder", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(household_id2, hh_s10q0a, hh_s10q01) %>%
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>%
  mutate(phone = `Fixed line telephone` + `Mobile Telephone`) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  mutate_at(vars(-("household_id2")),
            function(x) {ifelse(x>=1, 1, 0)}) %>%
  rename(radio=`Radio/tape recorder`, tv=Television, bike = Bicycle,
         motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

et3_ass %<>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio/tape recorder", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(household_id2, hh_s10q0a, hh_s10q01) %>%
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>%
  mutate(phone = `Fixed line telephone` + `Mobile Telephone`) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  mutate_at(vars(-("household_id2")),
            function(x) {ifelse(x>=1, 1, 0)}) %>%
  rename(radio=`Radio/tape recorder`, tv=Television, bike = Bicycle,
         motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

##*******************************************************************************
##### Consumption #####
##*******************************************************************************

et_cpi <- read.csv("../../Data/lsms/Ethiopia/eth_cpi.csv") %>% 
  filter(Country.Code == "ETH") %>% 
  select(Country.Code:X2021..YR2021.)

names(et_cpi) <- c("country",paste0("y_",2010:2021))

et_cpi %<>% 
  pivot_longer(cols = starts_with("y_"), names_to = "year", names_prefix = "y_", values_to = "yearly_cpi") %>% 
  mutate(deflator_2017 = 244.6490/yearly_cpi, 
         year = as.numeric(year)) %>% 
  select(year, deflator_2017)

et1_cons %<>% 
  select(household_id, rural, hh_size, adulteq, price_index_hce, total_cons_ann) %>%
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2011, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = (totcons_pc_adj * deflator_2017)/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326)) %>% 
  select(country, rural,household_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

et2_cons %<>% 
  select(household_id2, rural,hh_size, adulteq, price_index_hce, total_cons_ann) %>% 
  mutate(rural = ifelse(rural > 1,0,1)) %>% 
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2013, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = (totcons_pc_adj * deflator_2017)/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326)) %>% 
  select(country, rural,household_id2, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

et3_cons %<>% 
  select(household_id2, rural, hh_size, adulteq, price_index_hce, total_cons_ann) %>% 
  mutate(rural = ifelse(rural > 1,0,1)) %>% 
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2015, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = (totcons_pc_adj * deflator_2017)/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326)) %>% 
  select(country, rural,household_id2, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

#*******************************************************************************
#### merge data ####
#*******************************************************************************
et1 <- et1_house %>% 
  left_join(et1_ass, by = 'household_id') %>% 
  left_join(et1_cons, by = 'household_id') %>% 
  mutate(wave = 1) %>% 
  mutate(start_month = 09,
         start_year = 2011,
         end_month = 03,
         end_year = 2012)

et2 <- et2_house %>% 
  left_join(et2_ass, by = 'household_id2') %>% 
  left_join(et2_cons, by = 'household_id2') %>% 
  mutate(wave = 2) %>% 
  mutate(start_month = 09,
         start_year = 2013,
         end_month = 04,
         end_year = 2014)

et3 <- et3_house %>% 
  left_join(et3_ass, by = 'household_id2') %>% 
  left_join(et3_cons, by = 'household_id2') %>% 
  mutate(wave = 3) %>% 
  mutate(start_month = 09,
         start_year = 2015,
         end_month = 04,
         end_year = 2016)

#*******************************************************************************
#### split data ####
#*******************************************************************************
et1_eas = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/sect9_hh_w1.dta") %>% 
  select(household_id, ea_id)

long_panel <- rbind.data.frame(
  et1 %>% filter(household_id %in% long_panel_ids$household_id) %>% 
    left_join(long_panel_ids %>% select(household_id, household_id2), by = 'household_id') %>% 
    select(-household_id),
  et2 %>% filter(household_id2 %in% long_panel_ids$household_id2), 
  et3 %>% filter(household_id2 %in% long_panel_ids$household_id2)
)

long_panel %<>% left_join(long_panel_ids, by = 'household_id2') 

long_panel_attr <- et1 %>%
  filter((household_id %in% long_panel$household_id) == F) %>%
  left_join(et1_eas, by = 'household_id') %>% 
  left_join(long_panel_eas, by = 'ea_id')

short_panel <- rbind.data.frame(
  et2 %>% filter(household_id2 %in% short_panel_ids$household_id2), 
  et3 %>% filter(household_id2 %in% short_panel_ids$household_id2)
)

short_panel_attr <- et2 %>% 
  left_join(et2_geos %>% select(household_id2,ea_id2), by = 'household_id2') %>% 
  filter(ea_id2 %in% short_panel_ids$ea_id2) %>% 
  filter((household_id2 %in% short_panel$household_id2) == F)

#*******************************************************************************
#### final rearrange ####
#*******************************************************************************
long_panel %<>%
  mutate(case_id = paste0('eth_', household_id2),
         cluster_id = paste0('eth_', ea_id2)) %>% 
  relocate(country, start_month, start_year, end_month, end_year, wave, cluster_id, rural, lat, lon, case_id) %>% 
  select(-household_id2, -household_id, -ea_id, -ea_id2) %>% 
  mutate_all(as.vector)

long_panel_attr %<>% 
  mutate(case_id = paste0('eth_', household_id),
         cluster_id = paste0('eth_', ea_id)) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  relocate(country, start_month, start_year, end_month, end_year, wave, cluster_id, rural, lat, lon, case_id) %>% 
  select(-household_id, -ea_id, -ea_id2) %>% 
  mutate_all(as.vector)

short_panel %<>% 
  left_join(short_panel_ids, by = 'household_id2') %>% 
  mutate(case_id = paste0('eth_', household_id2),
         cluster_id = paste0('eth_', ea_id2)) %>% 
  relocate(country, start_month, start_year, end_month, end_year, wave, cluster_id, rural, lat, lon, case_id) %>% 
  select(-household_id2, -ea_id2) %>% 
  mutate_all(as.vector)

short_panel_attr %<>%
  left_join(short_panel_eas, by = 'ea_id2') %>% 
  mutate(case_id = paste0('eth_', household_id2),
         cluster_id = paste0('eth_', ea_id2)) %>%
  rename(lat = lat_2, lon = lon_2) %>% 
  relocate(country, start_month, start_year, end_month, end_year, wave, cluster_id, rural, lat, lon, case_id) %>% 
  select(-household_id2, -ea_id2) %>% 
  mutate_all(as.vector)


#*******************************************************************************
#### save data ####
#*******************************************************************************
write.csv(long_panel,"../../Data/processed/eth_labels_long.csv", row.names = F)
write.csv(long_panel_attr,"../../Data/processed/eth_labels_long_attr.csv", row.names = F)
write.csv(short_panel,"../../Data/processed/eth_labels_short.csv", row.names = F)
write.csv(short_panel_attr,"../../Data/processed/eth_labels_short_attr.csv", row.names = F)
