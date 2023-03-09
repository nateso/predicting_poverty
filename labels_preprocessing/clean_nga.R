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
##### Geo and ID Variables #####
#...............................................................................

# first create unique household id across all waves
ng1_geos %<>%
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>% 
  select(cluster_id, case_id, hhid, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_1 = lat_dd_mod, lon_1 = lon_dd_mod)

ng2_geos %<>% 
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
  select(cluster_id, case_id, hhid, LAT_DD_MOD, LON_DD_MOD) %>%
  rename(lat_2 = LAT_DD_MOD, lon_2 = LON_DD_MOD)

ng3_geos %<>% 
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
  select(cluster_id, case_id, hhid, LAT_DD_MOD, LON_DD_MOD) %>%
  rename(lat_3 = LAT_DD_MOD, lon_3= LON_DD_MOD)

ng4_id <- ng4_house %>% 
  mutate(cluster_id = paste0(lga,"_",ea)) %>% 
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>% 
  select(hhid, cluster_id, case_id)

ng4_geos %<>% 
  right_join(ng4_id, by = 'hhid') %>% 
  select(cluster_id, case_id, hhid, lat_dd_mod, lon_dd_mod) %>% 
  rename(lat_4 = lat_dd_mod, lon_4 = lon_dd_mod)


#### panel eas
ea_geos <- ng1_geos %>% 
  select(cluster_id, lat_1, lon_1) %>% 
  distinct()

long_panel_eas <- ng4_geos %>% 
  select(cluster_id) %>% distinct() %>% 
  filter(cluster_id %in% ea_geos$cluster_id) %>% 
  left_join(ea_geos, by = 'cluster_id')

short_panel_eas <- ea_geos %>% 
  filter((cluster_id %in% long_panel_eas$cluster_id) == F)


##### ID variables
w1_ids <- ng1_geos %>% 
  select(hhid, cluster_id, case_id) %>% 
  left_join(ng1_cons_1 %>% select(hhid, totcons), by = 'hhid') %>% 
  left_join(ng1_cons_2 %>% select(hhid, totcons), by = 'hhid', suffix = c("_1","_2")) %>% 
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons = ifelse(both_avail == 1, (totcons_1 + totcons_2)/2, 
                          ifelse(is.na(totcons_1), totcons_2, totcons_1))) %>% 
  filter(is.na(totcons) == F) %>% 
  select(case_id, cluster_id)

w2_ids <- ng2_geos %>% 
  filter(case_id %in% w1_ids$case_id) %>% # filter households that were not part of the first wave (e.g. due to missing consumption data)
  # the case_id also changes because it includes the ea, which is zero in case the household moved away...
  left_join(ng2_cons_1 %>% select(hhid, totcons), by = 'hhid') %>% 
  left_join(ng2_cons_2 %>% select(hhid, totcons), by = 'hhid', suffix = c("_1","_2")) %>% 
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons = ifelse(both_avail == 1, (totcons_1 + totcons_2)/2, 
                          ifelse(is.na(totcons_1), totcons_2, totcons_1))) %>% 
  filter(is.na(totcons) == F) %>% # filter households for which no consumption data is available
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(dist_y1y2 = distHaversine(cbind(lon_2,lat_2),cbind(lon_1,lat_1))) %>% 
  mutate(dist_y1y2 = ifelse(is.na(dist_y1y2),0,dist_y1y2)) %>% # assume that if no GPS data, household did not move (only one case)
  filter(dist_y1y2 < 10000) %>% # remove households that outside of EA (more than 10 kms away)
  select(case_id, cluster_id)
  
w3_ids <- ng3_geos %>% 
  filter(case_id %in% w2_ids$case_id) %>% # filter households that were not part of the first wave (e.g. due to missing consumption data)
  left_join(ng3_cons_1 %>% select(hhid, totcons), by = 'hhid') %>% 
  left_join(ng3_cons_2 %>% select(hhid, totcons), by = 'hhid', suffix = c("_1","_2")) %>% 
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons = ifelse(both_avail == 1, (totcons_1 + totcons_2)/2, 
                          ifelse(is.na(totcons_1), totcons_2, totcons_1))) %>% 
  filter(is.na(totcons) == F) %>% # filter households for which no consumption data is available
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(dist_y1y3 = distHaversine(cbind(lon_3,lat_3),cbind(lon_1,lat_1))) %>% 
  mutate(dist_y1y3 = ifelse(is.na(dist_y1y3),0,dist_y1y3)) %>% # assume that if no GPS data in wave 3, household did not move
  #filter(dist_y1y3 < 10000) %>% # remove households that outside of EA
    # do not remove, since this is only the case for two entire clusters (they slightly adjusted the coordinates for some reason...)
  select(case_id, cluster_id)
  
w4_ids <- ng4_geos %>% 
  filter(case_id %in% w3_ids$case_id) %>% # filter households that were not part of the first wave (e.g. due to missing consumption data)
  left_join(ng4_cons %>% select(hhid, totcons_pc), by = 'hhid') %>% 
  filter(is.na(totcons_pc) == F) %>% # filter households for which no consumption data is available
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(dist_y1y4 = distHaversine(cbind(lon_4,lat_4),cbind(lon_1,lat_1))) %>% 
  filter(dist_y1y4 < 10000) %>% 
  select(case_id, cluster_id)

## short panel ids
long_panel_ids <- w4_ids %>% 
  filter(case_id %in% w1_ids$case_id) %>% 
  filter(case_id %in% w2_ids$case_id) %>% 
  filter(case_id %in% w3_ids$case_id)

short_panel_ids <- w3_ids %>%
  filter(cluster_id %in% short_panel_eas$cluster_id)
#...............................................................................
##### Housing characteristics #####
#...............................................................................

ng1_house %<>%
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
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
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
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
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
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
  mutate(case_id = paste0(lga,"_",ea,"_",hhid)) %>%
  select(case_id, s11q6, s11q7, s11q8, s11q9, s11q43_1, s11q47, 
         s11q33b, s11q36) %>% 
  mutate(s11q47 = ifelse(s11q47 == 2,0,1)) %>% 
  rename(wall = s11q6, roof = s11q7, floor = s11q8, rooms=s11q9, 
         cooking_fuel = s11q43_1, electric=s11q47, watsup=s11q33b,
         toilet=s11q36) %>% 
  relocate(case_id, rooms, wall, roof, floor, electric, 
           watsup, toilet, cooking_fuel) ## phone is in asset data

#...............................................................................
###### Recode Housing ######
#...............................................................................
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

recode_files <- list.files("../../Data/lsms/Nigeria/recode/",full.names = T)
house_list <- list(ng1_house, ng2_house, ng3_house, ng4_house)
waves <- c('w1','w2','w3','w4')

for(i in 1:length(waves)){
  for(file in recode_files){
    file_name <- tail(strsplit(file,"/")[[1]],1)
    file_name <- sub(".csv","",file_name)
    if(grepl(waves[i],file_name)){
      recode_df <- read.csv(file)
      var_name <- unlist(strsplit(file_name,"_recode"))[1]
      house_list[[i]] <- house_list[[i]] %>% 
        left_join(recode_df, by = var_name, all.x =T)
      rm(var_name, recode_df)
    }
    rm(file_name)
  }
}

ng1_house <- house_list[[1]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ng2_house <- house_list[[2]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ng3_house <- house_list[[3]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ng4_house <- house_list[[4]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
 
#...............................................................................
##### Assets #####
#...............................................................................

assets = cbind.data.frame(code = c(322, 327, 312, 317, 318, 319, 3321, 3322),
                          label = c("radio",'tv','fridge','bike','motorcycle','car', 'smart_phone', 'mob_phone'))
ng1_ass %<>% select(hhid, item_cd, s5q1) 
hhids <- unique(ng1_ass$hhid)

# the asset dataset only includes those assets per hh of which the hh posses at least 1
for(id in hhids){
  aux <- ng1_ass %>% filter(hhid == id) %>% 
    filter(item_cd %in% assets$code)
  if(nrow(aux) == 0){
    no_ass <- data.frame(
      hhid = rep(id,6),
      item_cd = c(322, 327, 312, 317, 318, 319),
      s5q1 = rep(0,6)
    )
    ng1_ass <- rbind.data.frame(ng1_ass,no_ass)
    rm(no_ass)
  }
  rm(aux)
}

ng1_ass %<>% 
  filter(item_cd %in% assets$code) %>%
  left_join(assets, by = c('item_cd' = 'code')) %>% 
  mutate(s5q1 = ifelse(s5q1 > 0,1,0)) %>% 
  left_join(ng1_geos %>% select(case_id, hhid), by = 'hhid') %>%
  select(case_id,label,s5q1) %>% 
  pivot_wider(names_from = label, values_from = s5q1) %>% 
  replace(is.na(.), 0)# the asset dataset only includes those assets per hh of which the hh posses at least 1

ng2_ass %<>% 
  filter(item_cd %in% assets$code) %>%
  left_join(assets, by = c('item_cd' = 'code')) %>% 
  mutate(s5q1 = ifelse(s5q1 > 0,1,0)) %>% 
  left_join(ng2_geos %>% select(case_id, hhid), by = 'hhid') %>%
  select(case_id,label,s5q1) %>% 
  pivot_wider(names_from = label, values_from = s5q1)

ng3_ass %<>% 
  filter(item_cd %in% assets$code) %>%
  left_join(assets, by = c('item_cd' = 'code')) %>% 
  mutate(s5q1 = ifelse(s5q1 > 0,1,0)) %>% 
  left_join(ng3_geos %>% select(case_id, hhid), by = 'hhid') %>%
  select(case_id,label,s5q1) %>% 
  pivot_wider(names_from = label, values_from = s5q1)

ng4_ass %<>% 
  filter(item_cd %in% assets$code) %>%
  left_join(assets, by = c('item_cd' = 'code')) %>% 
  mutate(s5q1a = ifelse(s5q1a == 2,0,1)) %>% 
  left_join(ng4_geos %>% select(case_id, hhid), by = 'hhid') %>% 
  select(case_id,label,s5q1a) %>% 
  filter(is.na(case_id) == F) %>% 
  pivot_wider(names_from = label, values_from = s5q1a) %>%
  mutate(phone = ifelse(smart_phone + mob_phone > 0,1,0)) %>% 
  select(-smart_phone, -mob_phone)

#...............................................................................
##### Consumption data #####
#...............................................................................

# read in CPI data
ng_cpi <- read.csv("../../Data/lsms/Nigeria/cpi.csv") %>% 
  mutate(type = unlist(lapply(strsplit(indicator," "),function(x){x[1]})),
         month = unlist(lapply(strsplit(Date,"M"), function(x){x[2]})),
         year = as.numeric(unlist(lapply(strsplit(Date,"M"), function(x){x[1]})))) %>% 
  select(type,year,Value) %>% 
  rename(cpi = Value) %>% 
  group_by(type, year) %>% 
  summarise(yearly_cpi = sum(cpi)/12) %>% 
  ungroup() %>%
  filter(type != "Composite") %>% 
  mutate(urban = ifelse(type == "Urban",1,0)) %>% 
  mutate(deflator_2017 = ifelse(urban == 0, 231.65356/yearly_cpi, 235.55351/yearly_cpi)) %>% 
  select(-type, -yearly_cpi)

ng1_cons <- ng1_cons_1 %>% 
  left_join(ng1_geos %>% select(case_id, hhid), by = 'hhid') %>%
  mutate(year_1 = surveyprd, year_2 = surveyprd + 1) %>% 
  left_join(ng_cpi, by = c("year_1" = 'year', 'rururb' = 'urban')) %>% 
  left_join(ng_cpi, by = c("year_2" = 'year', 'rururb' = 'urban'), suffix = c("_1","_2")) %>% 
  left_join(ng1_cons_2 %>% select(hhid, totcons), by = 'hhid', suffix = c("_1","_2")) %>% 
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons_2017 = ifelse(both_avail == 1, (totcons_1 * deflator_2017_1 + totcons_2 * deflator_2017_2)/2, 
                               ifelse(is.na(totcons_1), totcons_2 * deflator_2017_2, totcons_1 * deflator_2017_1))) %>% 
  mutate(cons_pc_lcu_2017 = totcons_2017/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (112.098327636719)) %>% # converts the LCU into 2017 USD.
  mutate(rural = ifelse(rururb == 0,1,0),
         adulteq = NA) %>% 
  rename(hh_size = hhsize) %>% 
  select(case_id, rural, hh_size, adulteq, cons_pc_lcu_2017,cons_pc_usd_2017)

ng2_cons <- ng2_cons_2 %>% 
  left_join(ng2_geos %>% select(case_id, hhid), by = 'hhid') %>%
  mutate(year_1 = surveyprd, year_2 = surveyprd + 1) %>% 
  left_join(ng_cpi, by = c("year_1" = 'year', 'rururb' = 'urban')) %>% 
  left_join(ng_cpi, by = c("year_2" = 'year', 'rururb' = 'urban'), suffix = c("_1","_2")) %>% 
  left_join(ng2_cons_1 %>% select(hhid,totcons), by = 'hhid', suffix = c("_2","_1")) %>%
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons_2017 = ifelse(both_avail == 1, (totcons_1 * deflator_2017_1 + totcons_2 * deflator_2017_2)/2, 
                               ifelse(is.na(totcons_1), totcons_2 * deflator_2017_2, totcons_1 * deflator_2017_1))) %>% 
  mutate(cons_pc_lcu_2017 = totcons_2017/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (112.098327636719)) %>% # converts the LCU into 2017 USD.
  mutate(rural = ifelse(rururb == 0,1,0),
         adulteq = NA) %>% 
  rename(hh_size = hhsize) %>% 
  select(case_id, rural, hh_size, adulteq, cons_pc_lcu_2017,cons_pc_usd_2017)

ng3_cons <- ng3_cons_1 %>% 
  left_join(ng3_geos %>% select(case_id, hhid), by = 'hhid') %>%
  mutate(year_1 = surveyprd, year_2 = surveyprd + 1) %>% 
  left_join(ng_cpi, by = c("year_1" = 'year', 'rururb' = 'urban')) %>% 
  left_join(ng_cpi, by = c("year_2" = 'year', 'rururb' = 'urban'), suffix = c("_1","_2")) %>% 
  left_join(ng3_cons_2 %>% select(hhid,totcons), by = 'hhid', suffix = c("_1","_2")) %>%
  mutate(both_avail = ifelse(is.na(totcons_1) + is.na(totcons_2) == 0,1,0)) %>% 
  mutate(totcons_2017 = ifelse(both_avail == 1, (totcons_1 * deflator_2017_1 + totcons_2 * deflator_2017_2)/2, 
                          ifelse(is.na(totcons_1), totcons_2 * deflator_2017_2, totcons_1 * deflator_2017_1))) %>% 
  mutate(cons_pc_lcu_2017 = totcons_2017/365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (112.098327636719)) %>% # converts the LCU into 2017 USD.
  mutate(rural = ifelse(rururb == 0,1,0),
         adulteq = NA) %>% 
  rename(hh_size = hhsize) %>% 
  select(case_id, rural, hh_size, adulteq, cons_pc_lcu_2017,cons_pc_usd_2017)

ng4_cons <- ng4_cons %>% 
  left_join(ng4_geos %>% select(case_id, hhid), by = 'hhid') %>% 
  mutate(year = 2018, urban = ifelse(sector == 1,1,0)) %>% 
  left_join(ng_cpi, by = c('year' = 'year', 'urban' = 'urban')) %>% 
  mutate(yearly_cons_pc_lcu_2017 = totcons_pc * deflator_2017)  %>%
  mutate(cons_pc_lcu_2017 = yearly_cons_pc_lcu_2017 / 365) %>% 
  mutate(cons_pc_usd_2017 = cons_pc_lcu_2017 / (112.098327636719)) %>% # converts the LCU into 2017 USD.
  mutate(rural = ifelse(sector == 2,1,0),
         adulteq = NA) %>% 
  rename(hh_size = hhsize) %>% 
  select(case_id, rural, hh_size, adulteq, cons_pc_lcu_2017, cons_pc_usd_2017)

#...............................................................................
##### Merge Data per wave #####
#...............................................................................

ng1 <- ng1_geos %>% 
  select(case_id) %>% 
  left_join(ng1_house, by = 'case_id') %>% 
  left_join(ng1_ass, by = 'case_id') %>% 
  left_join(ng1_cons, by = 'case_id') %>% 
  mutate(country = 'nga',wave = 1) %>% 
  mutate(start_year = 2010, start_month = 08, end_year = 2011, end_month = 04)

ng2 <- ng2_geos %>% 
  select(case_id) %>% 
  left_join(ng2_house, by = 'case_id') %>% 
  left_join(ng2_ass, by = 'case_id') %>% 
  left_join(ng2_cons, by = 'case_id') %>% 
  mutate(country = 'nga', wave = 2) %>% 
  mutate(start_year = 2012, start_month = 09, end_year = 2013, end_month = 04) 

ng3 <- ng3_geos %>% 
  select(case_id) %>% 
  left_join(ng3_house, by = 'case_id') %>% 
  left_join(ng3_ass, by = 'case_id') %>% 
  left_join(ng3_cons, by = 'case_id') %>% 
  mutate(country = 'nga',wave = 3) %>% 
  mutate(start_year = 2015, start_month = 09, end_year = 2016, end_month = 04)
  
ng4 <- ng4_geos %>% 
  select(case_id) %>% 
  left_join(ng4_house, by = 'case_id') %>% 
  left_join(ng4_ass, by = 'case_id') %>% 
  left_join(ng4_cons, by = 'case_id') %>% 
  mutate(country = 'nga',wave = 4) %>% 
  mutate(start_year = 2018, start_month = 07, end_year = 2019, end_month = 02)


#...............................................................................
##### Split data removing migrant households #####
#...............................................................................

# split data into those households that were present in all 3 rounds and
# those households that moved to get an idea of the attrition rate
# and to assess whether the sample is biased due to attrition
# households that moved to a different EA are indicated by a 0 in the EA id
# thus the case_id changes as it also includes the EA id.

long_panel <- rbind.data.frame(
  ng1 %>% filter(case_id %in% long_panel_ids$case_id),
  ng2 %>% filter(case_id %in% long_panel_ids$case_id),
  ng3 %>% filter(case_id %in% long_panel_ids$case_id),
  ng4 %>% filter(case_id %in% long_panel_ids$case_id)
)

long_panel %<>% 
  left_join(long_panel_ids, by = 'case_id') %>% 
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)


short_panel <- rbind.data.frame(
  ng1 %>% filter(case_id %in% short_panel_ids$case_id),
  ng2 %>% filter(case_id %in% short_panel_ids$case_id),
  ng3 %>% filter(case_id %in% short_panel_ids$case_id)
)

short_panel %<>% 
  left_join(short_panel_ids, by = 'case_id') %>% 
  left_join(ea_geos, by = 'cluster_id') %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
           cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
           roof_qual,cooking_fuel_qual,
           toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
           hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

short_panel_attr <- ng1 %>% 
  left_join(ng1_geos %>% select(case_id, cluster_id), by = 'case_id') %>% 
  left_join(ea_geos, by = 'cluster_id') %>% 
  filter(cluster_id %in% short_panel_eas$cluster_id) %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  filter((case_id %in% short_panel$case_id) == F) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

long_panel_attr <- ng1 %>% 
  left_join(ng1_geos %>% select(case_id, cluster_id), by = 'case_id') %>% 
  left_join(ea_geos, by = 'cluster_id') %>% 
  filter(cluster_id %in% long_panel_eas$cluster_id) %>% 
  mutate(case_id = paste(country, case_id,sep = "_"),
         cluster_id = paste(country, cluster_id, sep = "_")) %>% 
  filter((case_id %in% long_panel$case_id) == F) %>% 
  rename(lat = lat_1, lon = lon_1) %>% 
  select(country,start_year, start_month, end_year, end_month, wave,
         cluster_id,rural,lat,lon,case_id,rooms,electric,floor_qual,wall_qual,
         roof_qual,cooking_fuel_qual,
         toilet_qual,watsup_qual,radio,tv,bike,motorcycle,fridge,car,phone,
         hh_size,adulteq,cons_pc_lcu_2017,cons_pc_usd_2017) %>% 
  mutate_all(as.vector)

# save data
write.csv(short_panel,"../../Data/processed/nga_short_labels.csv", row.names = F)
write.csv(long_panel,"../../Data/processed/nga_long_labels.csv", row.names = F)
write.csv(short_panel_attr, "../../Data/processed/nga_short_labels_attr.csv", row.names = F)
write.csv(long_panel_attr,"../../Data/processed/nga_long_labels_attr.csv", row.names =F)




















