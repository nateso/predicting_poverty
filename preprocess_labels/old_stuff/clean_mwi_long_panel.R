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
##### Geovariables and id variables #####
#...............................................................................

mw1_geos %<>% 
  select(case_id, lat_modified, lon_modified) %>%
  left_join(mw1_a %>% select(case_id, HHID), by = 'case_id') %>% 
  rename(lat_1 = lat_modified , lon_1 = lon_modified)

mw2_geos %<>% 
  select(y2_hhid, LAT_DD_MOD, LON_DD_MOD) %>%
  rename(lat_2 = LAT_DD_MOD, lon_2 = LON_DD_MOD)

mw3_geos %<>%
  select(y3_hhid, lat_modified, lon_modified) %>% 
  rename(lat_3 = lat_modified, lon_3 = lon_modified)

mw4_geos %<>%
  select(y4_hhid, lat_mod, lon_mod) %>% 
  rename(lat_4 = lat_mod, lon_4 = lon_mod)

# for the short panel

# short and long panel eas
long_ea_ids <- (mw3_a %>% select(ea_id) %>% distinct())$ea_id
short_ea_ids <- (mw1_a %>% select(ea_id) %>% 
  distinct() %>% filter((ea_id %in% long_ea_ids) == F))$ea_id


# long panel EAs and housheolds
# identify the 'mother' households. This assumes that suffixes indicate split off households
# in any case this makes sure that each household is represented by only one household per wave
long_ids_1<- mw1_a %>% select('case_id','HHID','ea_id') %>% 
  filter(ea_id %in% long_ea_ids) %>% 
  left_join(mw1_geos, by = 'case_id')

long_ids_2 <- mw2_a %>% select('case_id','y2_hhid','HHID','ea_id') %>% 
  filter(ea_id %in% long_ea_ids) %>% 
  arrange(HHID,y2_hhid) %>% 
  filter(!duplicated(HHID)) %>% 
  left_join(mw2_geos, by = 'y2_hhid')

long_ids_3 <- mw3_a %>% select('case_id','y2_hhid','HHID', 'y3_hhid','ea_id') %>% 
  filter(ea_id %in% long_ea_ids) %>% 
  filter(y2_hhid %in% long_ids_2$y2_hhid) %>% 
  arrange(HHID, y2_hhid, y3_hhid) %>% 
  filter(!duplicated(y2_hhid)) %>% 
  left_join(mw3_geos, by = 'y3_hhid')

long_ids_4 <- mw4_a %>% select('case_id','y3_hhid','HHID','y4_hhid','ea_id') %>% 
  filter(y3_hhid %in% long_ids_3$y3_hhid) %>% 
  arrange(HHID,y3_hhid,y4_hhid) %>% 
  filter(!duplicated(y3_hhid)) %>% 
  left_join(mw4_geos, by = 'y4_hhid')

long_main_ids <- (long_ids_4 %>% 
                    left_join(long_ids_1, by = 'case_id') %>% 
                    mutate(dist_14 = distHaversine(cbind(lon_4,lat_4),cbind(lon_1,lat_1))) %>% 
                    filter(dist_14 < 5000))$case_id # within 5km of the original EA

long_attr_ids <- (long_ids_1 %>% 
                    filter((case_id %in% long_main_ids) == F))$case_id


# short panel EAs and housheolds
short_ids_1 <- mw1_a %>% select('case_id','HHID','ea_id') %>% 
  filter(ea_id %in% short_ea_ids) %>% 
  left_join(mw1_geos, by = 'case_id')

short_ids_2 <- mw2_a %>% select('case_id','y2_hhid','HHID','ea_id') %>% 
  filter(ea_id %in% short_ea_ids) %>% 
  arrange(HHID,y2_hhid) %>% 
  filter(!duplicated(HHID)) %>% 
  left_join(mw2_geos, by = 'y2_hhid')

short_main_ids <- (short_ids_2 %>% 
                    left_join(short_ids_1, by = 'case_id') %>% 
                    mutate(dist_12 = distHaversine(cbind(lon_2,lat_2),cbind(lon_1,lat_1))) %>% 
                    filter(dist_12 < 5000))$case_id # within 5km of the original EA

short_attr_ids <- (short_ids_1 %>% 
                    filter((case_id %in% short_main_ids) == F))$case_id
  
#...............................................................................
##### Housing characteristics #####
#...............................................................................

mw1_house %<>%
  select(case_id, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw2_house %<>%
  select(y2_hhid, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw3_house %<>%
  select(y3_hhid, hh_f10, hh_f07, hh_f08, hh_f09, hh_f41, hh_f36, hh_f19, hh_f12,
         hh_f34) %>% 
  rename(rooms = hh_f10, wall = hh_f07, roof = hh_f08, floor=hh_f09, 
         toilet=hh_f41, watsup=hh_f36, electric=hh_f19, cooking_fuel = hh_f12, 
         phone = hh_f34) %>%
  mutate(electric=ifelse(electric==2, 0, 1),
         phone = ifelse(phone > 0,1,0))

mw4_house %<>%
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
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

recode_files <- list.files("../../Data/lsms/Malawi/recode/",full.names = T)
house_list <- list(mw1_house, mw2_house, mw3_house, mw4_house)
waves <- c('w1','w2','w3','w4')

for(file in recode_files){
  file_name <- tail(strsplit(file,"/")[[1]],1)
  file_name <- sub(".csv","",file_name)
  var_name <- unlist(strsplit(file_name,"_recode"))[1]
  recode_df <- read.csv(file)
  is_w4_file <- grepl("w4",file_name)
  if(is_w4_file){
    house_list[[4]] <- house_list[[4]] %>% 
      left_join(recode_df, by = var_name, all.x =T)
    next
  }
  for(i in 1:4){
    if(var_name %in% c('watsup','toilet') & i == 4){
      next
    }
    else{
      house_list[[i]] <- house_list[[i]] %>% 
        left_join(recode_df, by = var_name, all.x =T)
    }
  }
}


mw1_house <- house_list[[1]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
mw2_house <- house_list[[2]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
mw3_house <- house_list[[3]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
mw4_house <- house_list[[4]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
#...............................................................................
##### Assets #####
#...............................................................................
assets = cbind.data.frame(code = c(507,509,514,516,517,518),
                          label = c("radio",'tv','fridge','bike','motorcycle','car'))

mw1_ass %<>% 
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(case_id, label, hh_l01) %>%
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw2_ass  %<>% 
  select(y2_hhid, hh_l02, hh_l01) %>%
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y2_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw3_ass  %<>% 
  select(y3_hhid, hh_l02, hh_l01) %>%
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y3_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

mw4_ass  %<>% 
  select(y4_hhid, hh_l02, hh_l01) %>%
  filter(hh_l02 %in% assets$code) %>%
  left_join(assets, by = c('hh_l02' = 'code')) %>% 
  select(y4_hhid, label, hh_l01) %>% 
  mutate(hh_l01 = ifelse(hh_l01 == 2,0,1)) %>% 
  pivot_wider(names_from = label, values_from = hh_l01) 

#...............................................................................
##### Consumption #####
#...............................................................................
mwi_cpi <- read.csv("../../Data/lsms/Malawi/mwi_cpi.csv") %>% 
  filter(Country.Code == "MWI") %>% 
  select(-Series.Name, -Series.Code, -Country.Name, -X2021..YR2021.)

names(mwi_cpi) <- c("country",paste0("y_",2010:2020))

mwi_cpi %<>% 
  pivot_longer(cols = starts_with("y_"), names_to = "year", names_prefix = "y_", values_to = "yearly_cpi") %>% 
  mutate(deflator_2017 = 340.2421/yearly_cpi, 
         year = as.numeric(year)) %>% 
  select(year, deflator_2017)

mw1_cons %<>% 
  select(case_id, hhsize, adulteq, rexpagg, pcrexpagg, urban) %>% 
  mutate(cons_pc_lcu_2017 = (pcrexpagg * 2.0481145)/365)  %>%  # 2.0481145 is the value to inflate 2013 LCU to 2017 LCU.
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (241.930526733398)) %>% 
  mutate(wave = 1, year = 2010, country = 'mwi') %>% 
  rename(hh_size = hhsize) %>% 
  mutate(rural = ifelse(urban == 2,1,0)) %>% 
  select(country,year,wave, rural, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

mw2_cons %<>% 
  select(case_id, y2_hhid, hhsize, adulteq, rexpagg, pcrexpagg,urban) %>% 
  mutate(cons_pc_lcu_2017 = (pcrexpagg * 2.0481145)/365) %>%  # 2.0481145 is the value to inflate 2013 LCU to 2017 LCU.
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (241.930526733398)) %>% 
  mutate(wave = 2, year = 2013, country = 'mwi') %>% 
  rename(hh_size = hhsize) %>% 
  mutate(rural = ifelse(urban == 2,1,0)) %>% 
  select(country, year, wave, rural, y2_hhid, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)


# how to merge the data ????????? 

# problem: for IHPS 3 and IHPS4 (i.e. wave 3 and 4 of the panel, there is
# no aggregated consumption data available... would need to construct it myself...)
# aggregated consumption data is only available for the cross-sectional data...



#...............................................................................
##### merge data and split into attr and main data #####
#...............................................................................  

mw1 <- mw1_cons %>% 
  left_join(mw1_house, by = 'case_id') %>% 
  left_join(mw1_ass, by = 'case_id') %>% 
  mutate(start_year = 2010, start_month = 03, end_year = 2011, end_month = 03)

mw2 <- mw2_cons %>% 
  left_join(mw2_house, by = 'y2_hhid') %>% 
  left_join(mw2_ass, by = 'y2_hhid') %>% 
  mutate(start_year = 2013, start_month = 04, end_year = 12, end_month = 2013) %>% 
  filter(y2_hhid %in% long_ids_2$y2_hhid | y2_hhid %in% short_ids_2$y2_hhid) %>% # remove split off households!
  select(-y2_hhid) # case_id now uniquely identifies housheolds!

mw3 <- mw3_house %>% 
  left_join(mw3_ass, by = 'y3_hhid') %>% 
  filter(y3_hhid %in% long_ids_3$y3_hhid) %>% 
  mutate(wave = 3, year = 2016, country = 'mwi') %>% 
  mutate(start_year = 2016 , start_month = 04, end_year = 2017, end_month = 04) %>% 
  mutate(hh_size = NA, adulteq = NA, cons_pc_lcu_2017 = NA, cons_pc_usd_2017 = NA) %>% 
  left_join(long_ids_3 %>% select('y3_hhid','case_id'), by = 'y3_hhid') %>% 
  relocate(country,  start_month, start_year, end_month, end_year, wave, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017) %>% 
  select(-y3_hhid)

mw4 <- mw4_house %>% 
  left_join(mw4_ass, by = 'y4_hhid') %>% 
  filter(y4_hhid %in% long_ids_4$y4_hhid) %>% 
  mutate(wave = 4, year = 2019, country = 'mwi') %>% 
  mutate(start_year = 2019, start_month = 04, end_year = 2020, end_month = 04) %>% 
  mutate(hh_size = NA, adulteq = NA, cons_pc_lcu_2017 = NA, cons_pc_usd_2017 = NA) %>% 
  left_join(long_ids_4 %>% select('y4_hhid','case_id'), by = 'y4_hhid') %>% 
  relocate(country,  start_month, start_year, end_month, end_year, wave, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017) %>% 
  select(-y4_hhid)


mw_short_panel <- 
  rbind(mw1 %>% filter(case_id %in% short_main_ids),
        mw2 %>% filter(case_id %in% short_main_ids)) %>% 
  left_join(short_ids_1 %>% select(case_id,ea_id, lat_1, lon_1), by = 'case_id') %>% 
  mutate(case_id = paste0('mwi_', case_id),
         cluster_id = paste0('mwi_', ea_id)) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country, start_month, start_year, end_month, end_year, wave,
         cluster_id,rural,lat,lon,case_id, rooms,electric,floor_qual,
         wall_qual,roof_qual,cooking_fuel_qual,toilet_qual,watsup_qual,
         radio,tv,bike,motorcycle,fridge,car,phone, hh_size,adulteq,
         cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

mw_short_panel_attr <- # households may appear several times in the data as it includes households that moved...
  rbind(mw1 %>% filter(case_id %in% short_attr_ids),
        mw2 %>% filter(case_id %in% short_attr_ids)) %>% 
  left_join(short_ids_1 %>% select(case_id,ea_id, lat_1, lon_1), by = 'case_id') %>% 
  mutate(case_id = paste0('mwi_', case_id),
         cluster_id = paste0('mwi_', ea_id)) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country, start_month, start_year, end_month, end_year, wave,
         cluster_id,rural,lat,lon,case_id, rooms,electric,floor_qual,
         wall_qual,roof_qual,cooking_fuel_qual,toilet_qual,watsup_qual,
         radio,tv,bike,motorcycle,fridge,car,phone, hh_size,adulteq,
         cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)


mw_long_panel <- rbind(
  mw1 %>% filter(case_id %in% long_main_ids),
  mw2 %>% filter(case_id %in% long_main_ids),
  mw3 %>% filter(case_id %in% long_main_ids),
  mw4 %>% filter(case_id %in% long_main_ids)
  ) %>% 
  left_join(long_ids_1 %>% select(case_id, ea_id,lat_1, lon_1), by = 'case_id') %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  relocate(country, year, wave, ea_id, case_id, lat, lon)

mw_long_panel_attr <- rbind(
  mw1 %>% filter(case_id %in% long_attr_ids),
  mw2 %>% filter(case_id %in% long_attr_ids),
  mw3 %>% filter(case_id %in% long_attr_ids),
  mw4 %>% filter(case_id %in% long_attr_ids)
) %>% 
  left_join(long_ids_1 %>% select(case_id, ea_id,lat_1, lon_1), by = 'case_id') %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  relocate(country, year, wave, ea_id, case_id, lat, lon)


#...............................................................................
##### save data #####
#...............................................................................  
write.csv(mw_short_panel,"../../Data/processed/mwi_short_labels.csv", row.names = F)
write.csv(mw_short_panel_attr,"../../Data/processed/mwi_short_labels_attr.csv", row.names = F)
write.csv(mw_long_panel,"../../Data/processed/mwi_long_labels.csv", row.names = F)
write.csv(mw_long_panel_attr,"../../Data/processed/mwi_long_labels_attr.csv", row.names = F)


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


  



































