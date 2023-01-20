# Much of this code is inspired by the sustainBench package
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

et3_house = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Household/sect9_hh_w3.dta")
et3_ass = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Household/sect10_hh_w3.dta")
et3_cons = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/cons_agg_w3.dta")

et1_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2011/pub_eth_householdgeovariables_y1.dta")
et2_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2013/Pub_ETH_HouseholdGeovars_Y2.dta")
et3_geos = read_dta("../../Data/lsms/Ethiopia/first_ESS/ETH_2015/Geovariables/ETH_HouseholdGeovars_y3.dta")

#...............................................................................
##### ID variables #####
#...............................................................................

# In Ethiopia, the sample was extended to urban households in wave 2. Thus,
# the main id variable comes from wave 2

hh_ids_full <- et1_house %>% 
  select(household_id) %>%
  mutate(case_id = paste0("w1_",household_id)) %>% 
  left_join(et2_house %>% select(household_id, household_id2), by = 'household_id') %>% 
  mutate(household_id2 = ifelse(household_id2 == "",NA,household_id2))

hh_ids_sub <- et2_house %>%
  filter(household_id == '') %>% 
  select(household_id,household_id2) %>% 
  mutate(case_id = paste0('w2_',household_id2),
         household_id = NA)

all_hh_ids <- rbind.data.frame(hh_ids_full,hh_ids_sub)

# get the ids, which are subject to attrition
attr_w1w2w3 <- hh_ids_full %>% 
  filter((household_id %in% et3_house$household_id) == F |
           (household_id %in% et2_house$household_id) == F)

attr_w2w3 <- hh_ids_sub %>% 
  filter((household_id2 %in% et3_house$household_id2) == F)

attr_ids <- rbind.data.frame(attr_w1w2w3, attr_w2w3)

# get the main ids (either present in all 3 waves or only in 2 and 3)
# because of the additional EAs in wave 2
main_ids <- all_hh_ids %>% 
  filter((case_id %in% attr_ids$case_id) == F)


# get data that links every household to one ea inlcuding the geo vars
ea1_geos <- et1_geos %>% 
  select(ea_id, LAT_DD_MOD, LON_DD_MOD) %>% 
  rename(lat = LAT_DD_MOD, lon = LON_DD_MOD) %>% 
  mutate(cluster_id = paste0('w1_',ea_id)) %>% 
  select(cluster_id, lat, lon) %>% 
  distinct()

ea2_geos <- et2_geos %>% 
  filter(ea_id == "") %>% 
  rename(lat = lat_dd_mod, lon = lon_dd_mod) %>% 
  mutate(cluster_id = paste0('w2_',ea_id2)) %>% 
  select(cluster_id, lat, lon) %>% 
  distinct() %>% 
  drop_na()

ea3_geos <- et3_geos %>% 
  filter(ea_id2 %in% c("010501020100105","130101010100303")) %>% # for these two eas, geo vars become only available in wave 3 
  mutate(cluster_id = paste0('w2_',ea_id2)) %>% 
  select(cluster_id, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat = lat_dd_mod, lon = lon_dd_mod) %>% 
  distinct()

ea_geos <- rbind(ea1_geos, ea2_geos, ea3_geos)

hh_ea_ids <- all_hh_ids %>% 
  left_join(et1_house %>% select(household_id, ea_id), by = 'household_id') %>% 
  left_join(et2_house %>% select(household_id2, ea_id2), by = 'household_id2') %>% 
  mutate(cluster_id = ifelse(is.na(ea_id), "", paste0('w1_',ea_id))) %>% 
  mutate(cluster_id = ifelse(cluster_id == "", paste0('w2_',ea_id2), cluster_id)) %>% 
  select(case_id, cluster_id) %>% 
  left_join(ea_geos, by = 'cluster_id')


#...............................................................................
##### Housing characteristics #####
#...............................................................................

et1_house = et1_house %>%
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id') %>% 
  select(case_id, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0))

et2_house = et2_house %>%
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  select(case_id, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0)) 

et3_house = et3_house %>%
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id', suffix = c("","_w1")) %>%
  mutate(case_id = ifelse(is.na(case_id), case_id_w1, case_id)) %>% 
  filter(is.na(case_id) == F) %>% # some households only appear in wave 3 (just remove them...)
  select(case_id, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0)) 

#...............................................................................
###### Recode Housing ######
#...............................................................................
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
  left_join(watsup,by = c('watsup' = 'watsup_code')) %>% 
  select(-floor, -wall, -roof, -cooking_fuel, -toilet, -watsup)

#...............................................................................
##### Assets #####
#...............................................................................

et1_ass %<>% 
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id') %>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(case_id, hh_s10q0a, hh_s10q01) %>% 
  mutate(hh_s10q01 = ifelse(hh_s10q01 > 0,1,0)) %>% 
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>% 
  mutate(phone = ifelse(`Fixed line telephone` + `Mobile Telephone` > 0,1,0)) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  rename(radio=Radio, tv=Television, bike = Bicycle,
          motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

et2_ass %<>% 
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio/tape recorder", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(case_id, hh_s10q0a, hh_s10q01) %>%
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>%
  mutate(phone = `Fixed line telephone` + `Mobile Telephone`) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  mutate_at(vars(-("case_id")),
            function(x) {ifelse(x>=1, 1, 0)}) %>%
  rename(radio=`Radio/tape recorder`, tv=Television, bike = Bicycle,
         motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

et3_ass %<>% 
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id', suffix = c("","_w1")) %>%
  mutate(case_id = ifelse(is.na(case_id), case_id_w1, case_id)) %>% 
  filter(is.na(case_id) == F) %>% # some households only appear in wave 3 (just remove them...)
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio/tape recorder", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(case_id, hh_s10q0a, hh_s10q01) %>%
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>%
  mutate(phone = `Fixed line telephone` + `Mobile Telephone`) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  mutate_at(vars(-("case_id")),
            function(x) {ifelse(x>=1, 1, 0)}) %>%
  rename(radio=`Radio/tape recorder`, tv=Television, bike = Bicycle,
         motorcycle = Motorcycle, fridge=Refrigerator, car=`Private car`)

#...............................................................................
##### Consumption #####
#...............................................................................

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
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id') %>% 
  filter(is.na(total_cons_ann) == F) %>% # remove those households that did not report any consumption data
  select(case_id, hh_size, adulteq, price_index_hce, total_cons_ann) %>% 
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2011, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = totcons_pc_adj * deflator_2017) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326*365)) %>% 
  select(country, year, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

et2_cons %<>% 
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  filter(is.na(total_cons_ann) == F) %>% # remove those households that did not report any consumption data
  select(case_id, hh_size, adulteq, price_index_hce, total_cons_ann) %>% 
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2013, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = totcons_pc_adj * deflator_2017) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326*365)) %>% 
  select(country, year, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

et3_cons %<>% 
  left_join(all_hh_ids %>% select(household_id2, case_id), by = 'household_id2') %>% 
  left_join(all_hh_ids %>% select(household_id, case_id), by = 'household_id', suffix = c("","_w1")) %>%
  mutate(case_id = ifelse(is.na(case_id), case_id_w1, case_id)) %>% 
  filter(is.na(case_id) == F) %>% # some households only appear in wave 3 (just remove them...)
  filter(is.na(total_cons_ann) == F) %>% # remove those households that did not report any consumption data
  select(case_id, hh_size, adulteq, price_index_hce, total_cons_ann) %>% 
  mutate(totcons_pc_adj = total_cons_ann/hh_size * price_index_hce) %>% 
  mutate(year = 2015, country = 'eth') %>% 
  left_join(et_cpi, by = 'year') %>% 
  mutate(cons_pc_lcu_2017 = totcons_pc_adj * deflator_2017) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (8.49641704559326*365)) %>% 
  select(country, year, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

#...............................................................................
##### Combine data and add geovariables #####
#...............................................................................

et1 <- et1_cons %>% 
  left_join(et1_house, by = 'case_id') %>%
  left_join(et1_ass, by = 'case_id') %>% 
  mutate(wave = 1) %>% 
  mutate(across(everything(), as.vector))

et2 <- et2_cons %>% 
  left_join(et2_house, by = 'case_id') %>%
  left_join(et2_ass, by = 'case_id') %>% 
  mutate(wave = 2) %>% 
  mutate(across(everything(), as.vector))

et3 <- et3_cons %>% 
  left_join(et3_house, by = 'case_id') %>%
  left_join(et3_ass, by = 'case_id') %>% 
  mutate(wave = 3) %>% 
  mutate(across(everything(), as.vector))

et_attr <- rbind(et1,et2,et3) %>% 
  filter(case_id %in% attr_ids$case_id) %>%
  mutate(unique_id = paste(country,wave,case_id,sep = '_')) %>% 
  left_join(hh_ea_ids, by = 'case_id') %>% 
  relocate(country, year, wave, cluster_id, unique_id, case_id, lat, lon) %>% 
  filter(is.na(lat) == F) # remove the two clusters without 

et <- rbind(et1,et2,et3) %>% 
  filter((case_id %in% attr_ids$case_id) == F) %>%
  mutate(unique_id = paste(country,wave,case_id,sep = '_')) %>% 
  left_join(hh_ea_ids, by = 'case_id') %>% 
  relocate(country, year, wave, cluster_id, unique_id, case_id, lat, lon) 

# some households were removed due to missing values in either wave. Remove them from the other waves too.
in_w3 <- et %>% filter(wave == 3) %>% select(case_id)
in_w2 <- et %>% filter(wave == 2) %>% select(case_id)
et %<>% filter(case_id %in% in_w3$case_id) %>% filter(case_id %in% in_w2$case_id)

occur = et %>% group_by(case_id) %>% count()
table(et$wave)
table(occur$n) # numbers do match! -sweet! -finally!


#...............................................................................
##### save dataset #####
#...............................................................................

write.csv(et,"../../Data/processed/eth_labels.csv")
write.csv(et_attr, "../../Data/processed/eth_labels_attr.csv")






