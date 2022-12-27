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
#### NIGERIA ####
#*******************************************************************************

ng1_house <- read_dta("../../Data/lsms/Nigeria/NGA_2010/Post Harvest Wave 1/Household/sect8_harvestw1.dta")
ng1_ass <- read_dta("../../Data/lsms/Nigeria/NGA_2010/Post Planting Wave 1/Household/sect5_plantingw1.dta")
ng1_cons_1 <- read_dta('../../Data/lsms/Nigeria/NGA_2010/cons_agg_wave1_visit1.dta')
ng1_cons_2 <- read_dta('../../Data/lsms/Nigeria/NGA_2010/cons_agg_wave1_visit2.dta')
ng1_geos <- read_dta("../../Data/lsms/Nigeria/NGA_2010/Geodata/NGA_HouseholdGeovariables_Y1.dta")

ng2_house <- read_dta("../../Data/lsms/Nigeria/NGA_2012/Post Harvest Wave 2/Household/sect8_harvestw2.dta")
ng2_ass <- read_dta("../../Data/lsms/Nigeria/NGA_2012/Post Planting Wave 2/Household/sect5a_plantingw2.dta")
ng2_cons_1 <- read_dta("../../Data/lsms/Nigeria/NGA_2012/cons_agg_wave2_visit1.dta")
ng2_cons_2 <- read_dta("../../Data/lsms/Nigeria/NGA_2012/cons_agg_wave2_visit2.dta")
ng2_geos <- read_dta("../../Data/lsms/Nigeria/NGA_2012/Geodata Wave 2/NGA_HouseholdGeovars_Y2.dta")

ng3_house <- read_dta("../../Data/lsms/Nigeria/NGA_2015/sect11_plantingw3.dta")
ng3_ass <- read_dta("../../Data/lsms/Nigeria/NGA_2015/sect5_plantingw3.dta")
ng3_cons_1 <- read_dta("../../Data/lsms/Nigeria/NGA_2015/cons_agg_wave3_visit1.dta")
ng3_cons_2 <- read_dta("../../Data/lsms/Nigeria/NGA_2015/cons_agg_wave3_visit2.dta")
ng3_geos <- read_dta("../../Data/lsms/Nigeria/NGA_2015/NGA_HouseholdGeovars_Y3.dta")

ng4_house <- read_dta("../../Data/lsms/Nigeria/NGA_2018/sect11_plantingw4.dta")
ng4_ass <- read_dta("../../Data/lsms/Nigeria/NGA_2018/sect5_plantingw4.dta")
ng4_cons <- read_dta("../../Data/lsms/Nigeria/NGA_2018/totcons_final.dta")
ng4_geos <- read_dta("../../Data/lsms/Nigeria/NGA_2018/nga_householdgeovars_y4.dta")

#...............................................................................
##### Drop Migrant households #####
#...............................................................................
ea_geos <- ng1_geos %>% 
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  select(cluster_id, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat = lat_dd_mod, lon = lon_dd_mod) %>%
  distinct()

# first create unique household id across all waves
ng1_geos %<>%
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>% 
  select(cluster_id, case_id, hhid, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_1 = lat_dd_mod, lon_1 = lon_dd_mod)

ng2_geos %<>% 
  filter(ea != 0) %>% #remove households that moved (ea = 0)
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
  left_join(ea_geos,by = 'cluster_id') %>% 
  select(cluster_id, case_id, hhid, LAT_DD_MOD, LON_DD_MOD, lat, lon) %>%
  rename(lat_2 = LAT_DD_MOD, lon_2 = LON_DD_MOD) %>%
  mutate(dist_12 = distHaversine(cbind(lon_2,lat_2),cbind(lon,lat))) %>% 
  filter(dist_12 < 10000)

ng3_geos %<>% 
  filter(ea != 0) %>% #remove households that moved (ea = 0)
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
  left_join(ea_geos,by = 'cluster_id') %>% 
  select(cluster_id, case_id, hhid, LAT_DD_MOD, LON_DD_MOD, lat, lon) %>%
  rename(lat_3 = LAT_DD_MOD, lon_3= LON_DD_MOD) %>%
  mutate(dist_13 = distHaversine(cbind(lon_3,lat_3),cbind(lon,lat))) %>% 
  filter(dist_13 < 10000)

# get the EAs for which there are households in all three waves
ea_geos %<>%
  filter(cluster_id %in% ng2_geos$cluster_id) %>% 
  filter(cluster_id %in% ng3_geos$cluster_id)

# in wave 4 there was a sample refresh of the data, thus only parts of the
# sample are relevant. Filter those
ng4_id <- ng4_house %>% 
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>% 
  filter(case_id %in% ng1_geos$case_id) %>% 
  select(hhid, cluster_id, case_id)

ng4_geos %<>% 
  right_join(ng4_id, by = 'hhid') %>% 
  select(cluster_id, case_id, hhid, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_4 = lat_dd_mod, lon_4 = lon_dd_mod) %>% 
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(dist_14 = distHaversine(cbind(lon_4,lat_4),cbind(lon,lat))) %>% 
  filter(dist_14 < 10000) # still within 10 km of the original EA coords...

# get all relevant hosuehold ids
ng_ids <- ng1_geos %>%
  filter(cluster_id %in% ea_geos$cluster_id) %>% 
  select(cluster_id, case_id)

# Split-off households are not considered in the Nigeria panel data

#...............................................................................
##### Housing characteristics #####
#...............................................................................

ng1_house %<>%
  left_join(ng1_geos %>% select(case_id, hhid), by = 'hhid') %>%
  filter(case_id %in% ng_ids$case_id) %>% 
  select(case_id, s8q6, s8q7, s8q8, s8q9, s8q11, s8q17, s8q29, 
         s8q31, s8q33c, s8q36a) %>% 
  mutate(s8q17 = ifelse(s8q17 == 2,0,1), 
         s8q29 = ifelse(s8q29 == 2,0,1), 
         s8q31 = ifelse(s8q31 == 2,0,1)) %>% 
  mutate(phone = ifelse(s8q29 + s8q31 > 0,1,0)) %>% 
  rename(wall = s8q6, roof = s8q7, floor = s8q8, rooms=s8q9, 
         cooking_fuel = s8q11, electric=s8q17, watsup=s8q33c,
         toilet=s8q36a) %>% 
  select(-s8q29, -s8q31) %>%
  relocate(case_id, rooms, wall, roof, floor, electric, 
           watsup, toilet, cooking_fuel, phone)


ng2_house %<>% 
  left_join(ng2_geos %>% select(case_id, hhid), by = 'hhid') %>% 
  filter(case_id %in% ng_ids$case_id) %>% 
  select(case_id, s8q6, s8q7, s8q8, s8q9, s8q11, s8q17, s8q29, 
         s8q31, s8q33b, s8q36) %>% 
  mutate(s8q17 = ifelse(s8q17 == 2,0,1), 
         s8q29 = ifelse(s8q29 == 2,0,1), 
         s8q31 = ifelse(s8q31 == 2,0,1)) %>% 
  mutate(phone = ifelse(s8q29 + s8q31 > 0,1,0)) %>% 
  rename(wall = s8q6, roof = s8q7, floor = s8q8, rooms=s8q9, 
         cooking_fuel = s8q11, electric=s8q17, watsup=s8q33b,
         toilet=s8q36) %>% 
  select(-s8q29, -s8q31) %>%
  relocate(case_id, rooms, wall, roof, floor, electric, 
           watsup, toilet, cooking_fuel, phone)
    

ng3_house %<>% 
  left_join(ng3_geos %>% select(case_id, hhid), by = 'hhid') %>% 
  filter(case_id %in% ng_ids$case_id) %>% 
  select(case_id, s11q6, s11q7, s11q8, s11q9, s11q11, s11q17b, 
         s11q31, s11q33b, s11q36) %>% 
  mutate(s11q17b = ifelse(s11q17b == 2,0,1), 
         s11q31 = ifelse(s11q31 == 2,0,1)) %>% 
  rename(wall = s11q6, roof = s11q7, floor = s11q8, rooms=s11q9, 
         cooking_fuel = s11q11, electric=s11q17b, watsup=s11q33b,
         toilet=s11q36, phone = s11q31) %>% 
  relocate(case_id, rooms, wall, roof, floor, electric, 
           watsup, toilet, cooking_fuel, phone)


ng4_house %<>% 
  left_join(ng4_geos %>% select(case_id, hhid), by = 'hhid') %>% 
  filter(case_id %in% ng_ids$case_id) %>% 
  select(case_id, s11q6, s11q7, s11q8, s11q9, s11q43_1, s11q47, 
         s11q33b, s11q36) %>% 
  mutate(s11q47 = ifelse(s11q47 == 2,0,1)) %>% 
  rename(wall = s11q6, roof = s11q7, floor = s11q8, rooms=s11q9, 
         cooking_fuel = s11q43_1, electric=s11q47, watsup=s11q33b,
         toilet=s11q36) %>% 
  relocate(case_id, rooms, wall, roof, floor, electric, 
           watsup, toilet, cooking_fuel)

#...............................................................................
###### Recode Housing ######
#...............................................................................





    
        

  














