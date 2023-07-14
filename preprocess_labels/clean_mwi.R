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
#...............................................................................
##### Geovariables and id variables #####
#...............................................................................

mw1_geos %<>% 
  select(case_id, ea_id, lat_modified, lon_modified, reside) %>%
  left_join(mw1_a %>% select(case_id, HHID), by = 'case_id') %>% 
  rename(lat_1 = lat_modified , lon_1 = lon_modified) %>% 
  mutate(rural = ifelse(reside == 2,1,0)) %>% select(-reside)

mw2_geos %<>% 
  select(y2_hhid, LAT_DD_MOD, LON_DD_MOD) %>%
  rename(lat_2 = LAT_DD_MOD, lon_2 = LON_DD_MOD)

ea_geos <- mw1_geos %>% 
  select(ea_id, lat_1, lon_1, rural) %>% 
  distinct() %>% 
  rename(lat = lat_1, lon = lon_1)

w1_ids <- mw1_a %>% 
  select('case_id','ea_id') %>% 
  left_join(ea_geos, by = 'ea_id')

w2_ids <- mw2_a %>% 
  select(case_id, y2_hhid, HHID, ea_id) %>% 
  arrange(HHID, y2_hhid) %>% 
  filter(!duplicated(HHID)) %>% 
  left_join(ea_geos, by = 'ea_id') %>% 
  left_join(mw2_geos, by = 'y2_hhid') %>% 
  mutate(dist_12 = distHaversine(cbind(lon_2,lat_2),cbind(lon,lat))) %>% 
  filter(dist_12 < 10000) # within 10km of the original EA

short_main_ids <- w2_ids
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

#...............................................................................
###### Recode Housing ######
#...............................................................................
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

recode_files <- list.files("../../Data/lsms/Malawi/recode/",full.names = T)
house_list <- list(mw1_house, mw2_house)
waves <- c('w1','w2')

for(file in recode_files){
  file_name <- tail(strsplit(file,"/")[[1]],1)
  file_name <- sub(".csv","",file_name)
  is_w4_file <- grepl('w4',file_name)
  if(is_w4_file){next}
  var_name <- unlist(strsplit(file_name,"_recode"))[1]
  recode_df <- read.csv(file)
  for(i in 1:2){
    house_list[[i]] <- house_list[[i]] %>% 
        left_join(recode_df, by = var_name, all.x =T)
  }
}


mw1_house <- house_list[[1]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
mw2_house <- house_list[[2]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
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
  select(case_id, hhsize, adulteq, rexpagg, pcrexpagg) %>% 
  mutate(cons_pc_lcu_2017 = (pcrexpagg * 2.0481145)/365)  %>%  # 2.0481145 is the value to inflate 2013 LCU to 2017 LCU.
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (241.930526733398)) %>% 
  mutate(wave = 1, year = 2010, country = 'mwi') %>% 
  rename(hh_size = hhsize) %>% 
  select(country,year,wave, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

mw2_cons %<>% 
  select(case_id, y2_hhid, hhsize, adulteq, rexpagg, pcrexpagg) %>% 
  mutate(cons_pc_lcu_2017 = (pcrexpagg * 2.0481145)/365) %>%  # 2.0481145 is the value to inflate 2013 LCU to 2017 LCU.
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (241.930526733398)) %>% 
  mutate(wave = 2, year = 2013, country = 'mwi') %>% 
  rename(hh_size = hhsize) %>% 
  select(country, year, wave, y2_hhid, case_id, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

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
  mutate(start_year = 2013, start_month = 04, end_year = 2013, end_month = 12) %>% 
  filter(y2_hhid %in% short_main_ids$y2_hhid) %>% # remove split off households!
  select(-y2_hhid) # case_id now uniquely identifies housheolds!

mw_short_panel <- rbind.data.frame(
  mw1 %>% filter(case_id %in% short_main_ids$case_id),
  mw2 %>% filter(case_id %in% short_main_ids$case_id)
)

mw_short_panel %<>%  
  left_join(short_main_ids %>% select(case_id, ea_id), by = 'case_id') %>%
  left_join(ea_geos %>% select(ea_id, rural, lat, lon), by = 'ea_id') %>% 
  mutate(case_id = paste0('mwi_', case_id),
         cluster_id = paste0('mwi_', ea_id)) %>% 
  select(country, start_month, start_year, end_month, end_year, wave,
         cluster_id,rural,lat,lon,case_id, rooms,electric,floor_qual,
         wall_qual,roof_qual,cooking_fuel_qual,toilet_qual,watsup_qual,
         radio,tv,bike,motorcycle,fridge,car,phone, hh_size, adulteq,
         cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

mw_short_panel_attr <- mw1 %>% 
  left_join(short_main_ids %>% select(case_id, ea_id), by = 'case_id') %>%
  left_join(ea_geos %>% select(ea_id, rural, lat, lon), by = 'ea_id') %>%   
  mutate(case_id = paste0('mwi_', case_id),
         cluster_id = paste0('mwi_', ea_id)) %>% 
  filter((case_id %in% mw_short_panel$case_id) == F) %>% 
  select(country, start_month, start_year, end_month, end_year, wave,
         cluster_id,rural,lat,lon,case_id, rooms,electric,floor_qual,
         wall_qual,roof_qual,cooking_fuel_qual,toilet_qual,watsup_qual,
         radio,tv,bike,motorcycle,fridge,car,phone, hh_size, adulteq,
         cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

#...............................................................................
##### save data #####
#...............................................................................  
write.csv(mw_short_panel,"../../Data/lsms/processed/mwi_short_full_labels.csv", row.names = F)
write.csv(mw_short_panel_attr,"../../Data/lsms/processed/mwi_short_full_labels_attr.csv", row.names = F)




































