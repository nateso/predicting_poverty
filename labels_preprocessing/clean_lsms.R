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
##### Drop households #####
#...............................................................................

et1_geos <- et1_geos %>% 
  select(household_id,LAT_DD_MOD,LON_DD_MOD) %>% 
  filter(household_id %in% et1_ass$household_id) %>% 
  filter(!is.na(LAT_DD_MOD)) %>% 
  rename(lat_1 = LAT_DD_MOD, lon_1 = LON_DD_MOD)
  
et2_geos <- et2_geos %>% 
  select(household_id,household_id2,ea_id2,lat_dd_mod,lon_dd_mod) %>% 
  filter(household_id2 %in% et2_ass$household_id2) %>% 
  filter(!is.na(lat_dd_mod)) %>%
  rename(lat_2 = lat_dd_mod, lon_2 = lon_dd_mod)

et3_geos <- et3_geos %>% 
  select(household_id2,lat_dd_mod,lon_dd_mod) %>% 
  filter(household_id2 %in% et3_ass$household_id2) %>% 
  filter(!is.na(lat_dd_mod)) %>% 
  rename(lat_3 = lat_dd_mod, lon_3 = lon_dd_mod)
  
et_geos <- et2_geos %>% 
  left_join(et1_geos, by = 'household_id') %>% 
  left_join(et3_geos, by = 'household_id2') %>% 
  left_join(et2_house %>% select(household_id2, rural), by = 'household_id2') %>% 
  mutate(dist_12 = distHaversine(cbind(lon_1,lat_1), cbind(lon_2,lat_2)),
         dist_23 = distHaversine(cbind(lon_2,lat_2), cbind(lon_3,lat_3)))

# remove households if they moved from one wave to the other (i.e. dist12 > 0 or dist23 > 0)
# do not remove if distance is NA 
# remove if no coordinates available for wave 2
et_geos <- et_geos %>% 
  filter((dist_12 < 1) %>% replace_na(TRUE)) %>%
  filter((dist_23 < 1) %>% replace_na(TRUE)) %>% 
  filter(!is.na(lat_2)) %>% 
  select(household_id,household_id2,ea_id2,rural,lat_2,lon_2) %>% 
  rename(lat = lat_2, lon = lon_2)

# remove households from datasets and add household_id2 to wave 1 data

for(dat in ls()){
  if(grepl('1',dat)){
    aux <- get(dat) %>% filter(household_id %in% et_geos$household_id) %>%
      left_join(et_geos %>% select(household_id, household_id2), by = 'household_id')
    assign(dat, aux)
  }
  else{
    aux <- get(dat) %>% filter(household_id2 %in% et_geos$household_id2)
    assign(dat, aux)
  }
  rm(dat, aux)
}

#...............................................................................
##### Housing characteristics #####
#...............................................................................

et1_house = et1_house %>%
  select(household_id, household_id2, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0))

et2_house = et2_house %>%
  select(household_id, household_id2, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0)) 

et3_house = et3_house %>%
  select(household_id, household_id2, hh_s9q04, hh_s9q05, hh_s9q06, hh_s9q07, 
         hh_s9q10, hh_s9q13, hh_s9q19_a, hh_s9q21) %>%
  rename(rooms=hh_s9q04, wall = hh_s9q05, roof = hh_s9q06, floor=hh_s9q07, 
         toilet=hh_s9q10, watsup=hh_s9q13, electric=hh_s9q19_a, cooking_fuel = hh_s9q21) %>%
  mutate(electric=ifelse(electric<=4, 1, 0)) 

#...............................................................................
###### Recode Housing ######
#...............................................................................
source('lsms_utils.R')

floor = read.csv("../../Data/raw/Ethiopia/recode/floor_recode.csv")
wall = read.csv("../../Data/raw/Ethiopia/recode/wall_recode.csv")
roof = read.csv("../../Data/raw/Ethiopia/recode/roof_recode.csv")
cooking_fuel = read.csv("../../Data/raw/Ethiopia/recode/cooking_fuel_recode.csv")
toilet = read.csv("../../Data/raw/Ethiopia/recode/toilet_recode.csv")
watsup = read.csv("../../Data/raw/Ethiopia/recode/watsup_recode.csv")

et1_house %<>% merge_verbose(floor, by.x="floor", by.y="floor_code", all.x=T)
et1_house %<>% merge_verbose(wall, by.x="wall", by.y="wall_code", all.x=T)
et1_house %<>% merge_verbose(roof, by.x="roof", by.y="roof_code", all.x=T)
et1_house %<>% merge_verbose(cooking_fuel, by.x="cooking_fuel", by.y="cooking_fuel_code", all.x=T)
et1_house %<>% merge_verbose(toilet, by.x="toilet", by.y="toilet_code", all.x=T)
et1_house %<>% merge_verbose(watsup, by.x="watsup", by.y="watsup_code", all.x=T)

et2_house %<>% merge_verbose(floor, by.x="floor", by.y="floor_code", all.x=T)
et2_house %<>% merge_verbose(wall, by.x="wall", by.y="wall_code", all.x=T)
et2_house %<>% merge_verbose(roof, by.x="roof", by.y="roof_code", all.x=T)
et2_house %<>% merge_verbose(cooking_fuel, by.x="cooking_fuel", by.y="cooking_fuel_code", all.x=T)
et2_house %<>% merge_verbose(toilet, by.x="toilet", by.y="toilet_code", all.x=T)
et2_house %<>% merge_verbose(watsup, by.x="watsup", by.y="watsup_code", all.x=T)

# the variables for wave 3 change slightly. Thus adapt recode...
watsup = read.csv("../../Data/raw/Ethiopia/recode/watsup_recode_w3.csv")
toilet = read.csv("../../Data/raw/Ethiopia/recode/toilet_recode_w3.csv")

et3_house %<>% merge_verbose(floor, by.x="floor", by.y="floor_code", all.x=T)
et3_house %<>% merge_verbose(wall, by.x="wall", by.y="wall_code", all.x=T)
et3_house %<>% merge_verbose(roof, by.x="roof", by.y="roof_code", all.x=T)
et3_house %<>% merge_verbose(cooking_fuel, by.x="cooking_fuel", by.y="cooking_fuel_code", all.x=T)
et3_house %<>% merge_verbose(toilet, by.x="toilet", by.y="toilet_code", all.x=T)
et3_house %<>% merge_verbose(watsup, by.x="watsup", by.y="watsup_code", all.x=T)

#...............................................................................
##### Assets #####
#...............................................................................

et1_ass %<>% 
  filter(hh_s10q0a %in% c("Fixed line telephone", "Mobile Telephone",
                          "Radio", "Television", "Bicycle", "Motorcycle",
                          "Refrigerator", "Private car")) %>%
  select(household_id2, hh_s10q0a, hh_s10q01) %>%
  pivot_wider(names_from = hh_s10q0a, values_from = hh_s10q01) %>%
  mutate(phone = `Fixed line telephone` + `Mobile Telephone`) %>%
  select(-`Fixed line telephone`,-`Mobile Telephone`) %>% 
  mutate_at(vars(-("household_id2")),
            function(x) {ifelse(x>=1, 1, 0)}) %>%
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

#...............................................................................
##### Consumption #####
#...............................................................................

et1_cons %<>% 
  select(household_id2, price_index_hce, nom_totcons_aeq) %>% 
  mutate(year = 2011, country = 'et')

et2_cons %<>% 
  select(household_id2, price_index_hce, nom_totcons_aeq) %>%
  mutate(year = 2013, country = 'et')

et3_cons %<>% 
  select(household_id2, price_index_hce, nom_totcons_aeq) %>% 
  mutate(year = 2015, country = 'et')

#...............................................................................
##### Combine data #####
#...............................................................................

et1 <- et1_cons %>% 
  left_join(et1_house %>% select(-household_id), by = 'household_id2', suffix = c("","_y")) %>%
  left_join(et1_ass, by = 'household_id2', suffix = c("","_z")) %>% 
  mutate(across(everything(), as.vector))

et2 <- et2_cons %>% 
  left_join(et2_house %>% select(-household_id), by = 'household_id2', suffix = c("","_y")) %>%
  left_join(et2_ass, by = 'household_id2', suffix = c("","_z")) %>% 
  mutate(across(everything(), as.vector))

et3 <- et3_cons %>% 
  left_join(et3_house %>% select(-household_id), by = 'household_id2', suffix = c("","_y")) %>%
  left_join(et3_ass, by = 'household_id2', suffix = c("","_z")) %>% 
  mutate(across(everything(), as.vector))

et <- rbind(et1,et2,et3) %>% 
  left_join(et_geos %>% select(-household_id), by = 'household_id2', suffix = c("",'_g')) %>% 
  mutate(across(everything(), as.vector)) %>% 
  mutate(clusterid = paste0(country,"_",ea_id2)) %>% 
  select(-ea_id2) %>% 
  relocate(country,year,clusterid,rural,lat,lon) %>% 
  mutate(rural = ifelse(rural > 1,0,1)) # check other countries if difference between small and large town.

write.csv(et,"../../Data/processed/eth_labels.csv")






