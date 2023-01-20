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
#### MALAWI ####
#*******************************************************************************

mw1_a = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_A_FILT_10.dta")
mw1_house = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_F_10.dta") 
mw1_ass = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_L_10.dta")
mw1_cons = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/Round 1 (2010) Consumption Aggregate.dta")
mw1_geos = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HouseholdGeovariables_IHS3_Rerelease_10.dta")

mw2_a = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_A_FILT_13.dta")
mw2_house = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_F_13.dta") 
mw2_ass = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HH_MOD_L_13.dta")
mw2_cons = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/Round 2 (2013) Consumption Aggregate.dta")
mw2_geos = read_dta("../../Data/lsms/Malawi/MWI_2010_2013/HouseholdGeovariables_IHPS_13.dta")

mw3_a = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_a_filt_16.dta")
mw3_house = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_f_16.dta") 
mw3_ass = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_l_16.dta") 
mw3_cons = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/ihs4 consumption aggregate.dta")
mw3_geos = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/HouseholdGeovariablesIHPSY3.dta")

mw4_a = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_a_filt_19.dta")
mw4_house = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_f_19.dta") 
mw4_ass = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_mod_l_19.dta")
mw4_cons = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/ihs5_consumption_aggregate.dta")
mw4_geos = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/householdgeovariables_y4.dta")

#...............................................................................
##### Drop Migrant households #####
#...............................................................................

mw1_geos %<>% 
  select(case_id, lat_modified, lon_modified) %>%
  left_join(mw1_a %>% select(case_id, HHID), by = 'case_id') %>% 
  rename(lat_1 = lat_modified , lon_1 = lon_modified)

mw2_geos %<>% 
  left_join(mw2_a %>% select(y2_hhid,case_id), by = 'y2_hhid') %>% 
  select(case_id, y2_hhid, LAT_DD_MOD, LON_DD_MOD) %>%
  rename(lat_2 = LAT_DD_MOD, lon_2 = LON_DD_MOD)

mw3_geos %<>%
  left_join(mw3_a %>% select(y3_hhid, y2_hhid, case_id), by = 'y3_hhid') %>% 
  select(case_id, y3_hhid, y2_hhid, lat_modified, lon_modified) %>% 
  rename(lat_3 = lat_modified, lon_3 = lon_modified)

mw4_geos %<>%
  left_join(mw4_a %>% select(y4_hhid, y3_hhid, case_id), by = 'y4_hhid') %>%
  select(case_id, y4_hhid, y3_hhid, lat_mod, lon_mod) %>% 
  rename(lat_4 = lat_mod, lon_4 = lon_mod)

# for the short panel
mw_ids_short <- mw2_a %>% 
  left_join(mw2_geos %>% select(y2_hhid, lat_2, lon_2), by = 'y2_hhid') %>%
  left_join(mw1_geos %>% select(HHID, lat_1, lon_1), by = 'HHID') %>% 
  mutate(dist_12 = distHaversine(cbind(lon_1,lat_1), cbind(lon_2,lat_2))) %>% 
  filter(dist_12 < 1) %>%  # remove households that have moved to a different EA
  #filter(hh_a05 == 1) %>% # remove split off households
  select(case_id, HHID, y2_hhid, lat_1, lon_1) # lat_1 == lat_2 

mw_ids_long <- mw_ids_short %>% 
  left_join(mw3_geos %>% select(y2_hhid, y3_hhid, lat_3, lon_3), by = 'y2_hhid') %>% 
  mutate(dist_13 = distHaversine(cbind(lon_1,lat_1),cbind(lon_3, lat_3))) %>%
  filter(dist_13 < 1) %>% # remove households that moved to different EA
  left_join(mw4_geos %>% select(y3_hhid, y4_hhid, lat_4, lon_4), by = 'y3_hhid') %>% 
  mutate(dist_14 = distHaversine(cbind(lon_1,lat_1),cbind(lon_4, lat_4))) %>% 
  filter(dist_14 < 1) %>%
  select(case_id, HHID, y2_hhid, y3_hhid, y4_hhid, lat_1, lon_1) %>% 
  rename(lat = lat_1, lon = lon_1)
  
#...............................................................................
##### Housing characteristics #####
#...............................................................................

mw1_house %<>%
  filter(case_id %in% mw_ids_short$case_id) %>% 
  select(case_id, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw2_house %<>%
  filter(y2_hhid %in% mw_ids_short$y2_hhid) %>% 
  select(y2_hhid, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw3_house %<>%
  filter(y3_hhid %in% mw_ids_long$y3_hhid) %>% 
  select(y3_hhid, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw4_house %<>%
  filter(y4_hhid %in% mw_ids_long$y4_hhid) %>% 
  select(y4_hhid, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

#...............................................................................
###### Recode Housing ######
#...............................................................................



#...............................................................................
##### Assets #####
#...............................................................................
assets = cbind.data.frame(code = c(507,509,514,516,517,518),
                          label = c("radio",'tv','fridge','bike','motorcycle','car'))

mw1_ass %<>% 
  filter(case_id %in% mw_ids_short$case_id) %>% 
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(case_id, label, hh_l01) %>%
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw2_ass  %<>% 
  select(y2_hhid, hh_l02, hh_l01) %>%
  filter(y2_hhid %in% mw_ids_short$y2_hhid) %>% 
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y2_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw3_ass  %<>% 
  select(y3_hhid, hh_l02, hh_l01) %>%
  filter(y3_hhid %in% mw_ids_long$y3_hhid) %>% 
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y3_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw4_ass  %<>% 
  select(y4_hhid, hh_l02, hh_l01) %>%
  filter(y4_hhid %in% mw_ids_long$y4_hhid) %>% 
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y4_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

#...............................................................................
##### Consumption #####
#...............................................................................
mw1_cons %<>% 
  filter(case_id %in% mw_ids_short$case_id) %>% 
  select(case_id, pcrexpagg) %>% 
  rename(real_cons_2013_pc = pcrexpagg)

mw2_cons %<>%
  filter(y2_hhid %in% mw_ids_short$y2_hhid) %>% 
  select(y2_hhid, pcrexpagg) %>% 
  rename(real_cons_2013_pc = pcrexpagg)

# how to merge the data ????????? 

# problem: for IHPS 3 and IHPS4 (i.e. wave 3 and 4 of the panel, there is
# no aggregated consumption data available... would need to construct it myself...)
# aggregated consumption data is only available for the cross-sectional data...
mw3_cons %<>%
  right_join(mw3_house %>% select())
  filter(y3_hhid %in% mw_ids_long$y3_hhid) %>% 
  select(y3_hhid, rexpaggpc) %>% 
  rename(real_cons_2016_pc = rexpaggpc)

mw4_cons %<>%
  filter(y4_hhid %in% mw_ids_long$y4_hhid) %>% 
  select(y4_hhid, rexpaggpc) %>% 
  rename(real_cons_2019_pc = rexpaggpc)
  



# 
# 
# mw3_meta_panel = read_dta("../../Data/lsms/Malawi/MWI_2010_2019/hh_meta_16.dta")
# mw3_meta_cross = read_dta("../../Data/lsms/Malawi/MWI_2016/household/hh_metadata.dta")
# mw3_geo_cross = read_dta("../../Data/lsms/Malawi/MWI_2016/household_geovariables/householdgeovariablesihs4.dta")
# 
# mw3_meta_panel %<>% 
#   filter(y3_hhid %in% mw_ids_long$y3_hhid) %>% 
#   mutate(id = paste0(
#     as.character(moduleB_start_date),"_",
#     as.character(moduleB_startHr),"_",
#     as.character(moduleB_startMin),"_",
#     as.character(moduleF_startHr),"_",
#     as.character(moduleF_startMin),"_",
#     as.character(moduleV_startHr),"_",
#     as.character(moduleV_startMin),"_",
#     as.character(moduleX_startHr),"_",
#     as.character(moduleX_startMin)))
# 
# mw3_meta_cross %<>% mutate(
#   id = paste0(
#     as.character(moduleB_start_date),"_",
#     as.character(moduleB_startHr),"_",
#     as.character(moduleB_startMin),"_",
#     as.character(moduleF_startHr),"_",
#     as.character(moduleF_startMin),"_",
#     as.character(moduleV_startHr),"_",
#     as.character(moduleV_startMin),"_",
#     as.character(moduleX_startHr),"_",
#     as.character(moduleX_startMin)))
# 
# mw3_meta_cross %<>% 
#   filter(id %in% mw3_meta_panel$id)
# 
# mw3_geo_cross %<>% 
#   filter(!duplicated(lat_modified)) %>% 
#   filter(!is.na(lat_modified))
# 
# is_in = rep(NA,nrow(mw3_geo_cross))
# for (i in 1:nrow(mw3_geo_cross)){
#   lat = mw3_geo_cross$lat_modified[i]
#   lon = mw3_geo_cross$lon_modified[i]
#   dists = distHaversine(c(lon,lat), cbind(mw_ids_long$lon, mw_ids_long$lat))
#   if (min(dists) > 1000){
#     is_in[i] = F
#   }
#   else{
#     is_in[i] = T
#   }
# }


  



































