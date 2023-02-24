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
#### UGANDA ####
#*******************************************************************************

# just to check how many households were sampled at the very beginning (2005 - no aggregated consumption data available)
ug0_a <- read_dta("../../Data/lsms/Uganda/UGA_2005/2005/2005_GSEC1.dta")

ug1_house = read_dta("../../Data/lsms/Uganda/UGA_2009/2009_GSEC9.dta")
ug1_ass = read_dta("../../Data/lsms/Uganda/UGA_2009/2009_GSEC14.dta")
ug1_cons = read_dta("../../Data/lsms/Uganda/UGA_2009/pov2009_10.dta")
ug1_geos = read_dta("../../Data/lsms/Uganda/UGA_2009/2009_UNPS_Geovars_0910.dta") 
ug1_a = read_dta("../../Data/lsms/Uganda/UGA_2009/2009_GSEC1.dta")

ug2_house = read_dta("../../Data/lsms/Uganda/UGA_2010/GSEC9A.dta")
ug2_ass = read_dta("../../Data/lsms/Uganda/UGA_2010/GSEC14.dta")
ug2_cons = read_dta("../../Data/lsms/Uganda/UGA_2010/pov2010_11.dta")
ug2_geos = read_dta("../../Data/lsms/Uganda/UGA_2010/UNPS_Geovars_1011.dta") 
ug2_a = read_dta("../../Data/lsms/Uganda/UGA_2010/GSEC1.dta")

ug3_house = read_dta("../../Data/lsms/Uganda/UGA_2011/GSEC9A.dta")
ug3_ass = read_dta("../../Data/lsms/Uganda/UGA_2011/GSEC14.dta")
ug3_cons = read_dta("../../Data/lsms/Uganda/UGA_2011/UNPS 2011-12 Consumption Aggregate.dta")
ug3_geos = read_dta("../../Data/lsms/Uganda/UGA_2011/UNPS_Geovars_1112.dta") 
ug3_a = read_dta("../../Data/lsms/Uganda/UGA_2011/GSEC1.dta")

# sample refresh in wave 4 - 1/3 of the sample was replaced by new households
# no GPS data starting wave 4, thus no GPS data for new enumeration areas...
ug4_house = read_dta("../../Data/lsms/Uganda/UGA_2013/GSEC9_1.dta")
ug4_ass = read_dta("../../Data/lsms/Uganda/UGA_2013/GSEC14A.dta")
ug4_cons = read_dta("../../Data/lsms/Uganda/UGA_2013/pov2013_14.dta")
ug4_a = read_dta("../../Data/lsms/Uganda/UGA_2013/GSEC1.dta") # GPS data not available for wave 4

ug5_house = read_dta("../../Data/lsms/Uganda/UGA_2015/gsec9.dta")
ug5_ass = read_dta("../../Data/lsms/Uganda/UGA_2015/gsec14.dta")
ug5_cons = read_dta("../../Data/lsms/Uganda/UGA_2015/pov2015_16.dta")
ug5_a = read_dta("../../Data/lsms/Uganda/UGA_2015/gsec1.dta") # GPS data not available for wave 5

ug6_house = read_dta("../../Data/lsms/Uganda/UGA_2018/HH/GSEC9.dta")
ug6_ass = read_dta("../../Data/lsms/Uganda/UGA_2018/HH/GSEC14.dta")
ug6_cons = read_dta("../../Data/lsms/Uganda/UGA_2018/pov2018_19.dta")
ug6_a = read_dta("../../Data/lsms/Uganda/UGA_2018/HH/GSEC1.dta") # GPS data not available for wave 6

ug7_house = read_dta("../../Data/lsms/Uganda/UGA_2019/HH/gsec9.dta")
ug7_ass = read_dta("../../Data/lsms/Uganda/UGA_2019/HH/gsec14.dta")
ug7_cons = read_dta("../../Data/lsms/Uganda/UGA_2019/pov2019_20.dta")
ug7_a = read_dta("../../Data/lsms/Uganda/UGA_2019/HH/gsec1.dta") # GPS data not available for wave 7

#*******************************************************************************
#### id variables ####
#*******************************************************************************
#* ISSUES:
#* 1. Attrition (household might not appear in later survey rounds) 
#* 2. Household might not appear in a survey round, but in the subsequent it does again...
#* 3. Data includes split-off households (remove those)
#* 4. Households that moved are included in the data (within and outside of EA.)

# Geovariables and cluster ids
ug_geos <- ug1_geos %>% 
  filter(hh_status < 3,
         is.na(dist_y0)) %>% # if dist_y0 is positive, it is considered that the household moved outside of the EA. (more than 5km)
  left_join(ug0_a, by = c("HHID"='Hhid'),suffix = c("","_w0")) %>% 
  select(HHID,lat_mod,lon_mod,Comm) %>% 
  rename(w1_hhid = HHID, cluster_id = Comm, lat = lat_mod, lon = lon_mod)

hhid_comm <- ug_geos %>% select(w1_hhid,cluster_id) %>% mutate_all(as.vector)

ug_geos %<>% drop_na() %>% # drop entry if any lat or lon is na
  select(-w1_hhid) %>% distinct() # one pair of coordinates for each cluster.

###### get ids for each round (removing split-offs and checking whether household moved.)

# wave 1 / baseline households
w1_ids <- hhid_comm %>% select(w1_hhid) %>% 
  filter(w1_hhid %in% ug1_cons$hhid)

# wave 2 
w2_ids <- ug2_a %>% 
  filter(HHID %in% w1_ids$w1_hhid) %>% # remove split-offs 
  left_join(ug2_geos %>% select(HHID,dist_y0), by = 'HHID') %>% 
  mutate(moved_EA = ifelse(is.na(dist_y0),0,1)) %>% 
  filter(moved_EA == 0) %>% # remove households that moved
  select(HHID) %>% 
  rename(w1_hhid = HHID) %>% 
  filter(w1_hhid %in% ug2_cons$hh)
  
# wave 3 
w3_ids <- ug3_a %>% 
  filter(HHID %in% w2_ids$w1_hhid) %>% # remove split-offs
  left_join(ug3_geos %>% select(HHID,dist_y0), by = 'HHID') %>% 
  mutate(moved_EA = ifelse(is.na(dist_y0),0,1)) %>% 
  filter(moved_EA == 0) %>% # remove households that moved
  select(HHID) %>% 
  rename(w1_hhid = HHID) %>% 
  filter(w1_hhid %in% ug3_cons$HHID)

# Starting in wave 4, GPS data is no longer available
# 1/3 of the sample was refreshed in wave 4 - however no GPS data available for the new EAs

# wave 4 - remove split-off households
w4_ids <- ug4_a %>% 
  filter(rotate == 1) %>% # 0 indicates brand-new households, 2 indicates split-offs (but no new split-offs)
  mutate(HHID_old = as.character(HHID_old)) %>% 
  filter(is.na(HHID_old) == F) %>% # Remove households that have no old HHID (split-offs? or households that moved?)
  filter(HHID_old %in% w3_ids$w1_hhid) %>% # remove split-off households from earlier periods
  select(HHID,HHID_old) %>% 
  rename(w4_hhid=HHID, w1_hhid = HHID_old) %>% 
  filter(w4_hhid %in% ug4_cons$HHID)

# wave 5 - remove split-off households
ug5_a %<>%
  mutate(HHID_old = unlist(lapply(ug5_a$hh,function(x){gsub("-05-","-04-",x)})))

w5_ids <- ug5_a %>% 
  filter(HHID_old %in% w4_ids$w4_hhid) %>% 
  select(HHID,hh,HHID_old) %>% 
  rename(w5_hhid=HHID, w5_hh = hh, w4_hhid = HHID_old) %>% 
  filter(w5_hh %in% ug5_cons$hh)

# wave 6 - remove split-off households
w6_ids <- ug6_a %>% 
  filter(t0_hhid %in% w5_ids$w5_hhid) %>% 
  select(hhid,t0_hhid) %>% 
  rename(w6_hhid=hhid, w5_hhid = t0_hhid) %>% 
  filter(w6_hhid %in% ug6_cons$hhid)

# wave 7 - remove split-off households
w7_ids <- ug7_a %>% 
  filter(hhidold != "") %>% 
  filter(hhidold %in% w6_ids$w6_hhid) %>% 
  select(hhid,hhidold) %>% 
  rename('w7_hhid'=hhid, w6_hhid = hhidold) %>% 
  filter(w7_hhid %in% ug7_cons$hhid[ug7_cons$cpexp30 > 0]) # some report consumption = 0

#### check which EAs are long-panel EAs and which ones belong to the short panel only
eas <- unique(hhid_comm$cluster_id) 
long_eas <- w4_ids %>% 
  left_join(hhid_comm, by = "w1_hhid") 
long_eas <- unique(long_eas$cluster_id)
short_eas <- eas[(eas %in% long_eas) == F]

short_panel_ids <- w3_ids %>% 
  left_join(hhid_comm,by = 'w1_hhid') %>% 
  filter(cluster_id %in% short_eas)

long_panel_ids <- w7_ids %>% 
  left_join(w6_ids, by = 'w6_hhid') %>%
  left_join(w5_ids, by = 'w5_hhid') %>% 
  left_join(w4_ids, by = 'w4_hhid') %>% 
  left_join(w3_ids, by = 'w1_hhid') %>% 
  left_join(hhid_comm, by = 'w1_hhid') %>% 
  filter(cluster_id %in% long_eas)

#...............................................................................
##### Housing characteristics #####
#...............................................................................

extract_housing <- function(dat,id_var,rm,wl,rf,fl,toi,wat){
  vars <- c(id_var, rm,wl,rf,fl,toi,wat)
  aux <- dat %>%
    select(all_of(vars)) %>% 
    rename(rooms = rm, wall = wl, roof = rf, floor=fl, 
           toilet=toi, watsup=wat) 
  return(aux)
}


ug1_house <- extract_housing(ug1_house,'Hhid','H9q03','H9q05','H9q04','H9q06','H9q22','H9q07') %>% 
  rename(HHID = Hhid)
ug2_house <- extract_housing(ug2_house,'HHID','h9q3','h9q5','h9q4','h9q6','h9q22','h9q7')
ug3_house <- extract_housing(ug3_house,'HHID','h9q3','h9q5','h9q4','h9q6','h9q22','h9q7')
ug4_house <- extract_housing(ug4_house,'HHID','h9q3','h9q5','h9q4','h9q6','h9q22','h9q7')
ug5_house <- extract_housing(ug5_house,'hhid','h9q3','h9q5','h9q4','h9q6','h9q22','h9q7')
ug6_house <- extract_housing(ug6_house,'hhid','h9q03','h9q05','h9q04','h9q06','h9q22','h9q07')
ug7_house <- extract_housing(ug7_house,'hhid','h9q03','h9q05','h9q04','h9q06','h9q22','h9q07')


# electricity and fuel are in a different questionnaire
extract_electric <- function(dat, id_var, path, elec_var, fuel_var){
  elec <- read_dta(path) %>% 
    rename('electric' = elec_var,
           'cooking_fuel' = fuel_var) %>% 
    mutate(electric = ifelse(electric == 2,0,1)) %>% 
    select(id_var,electric,cooking_fuel)
  aux <- dat %>% left_join(elec,by = id_var) %>%  mutate_all(as.vector)
  return(aux)
}

path_elec <- "../../Data/lsms/Uganda/UGA_2009/2009_GSEC10A.dta"
ug1_house <- extract_electric(ug1_house,'HHID',path_elec, 'h10q1', 'h10q09')

path_elec <- "../../Data/lsms/Uganda/UGA_2010/GSEC10A.dta"
ug2_house <- extract_electric(ug2_house,'HHID',path_elec, 'h10q1','h10q9')

path_elec <- "../../Data/lsms/Uganda/UGA_2011/GSEC10A.dta"
ug3_house <- extract_electric(ug3_house,'HHID',path_elec, 'h10q1','h10q9')

path_elec <- "../../Data/lsms/Uganda/UGA_2013/GSEC10_1.dta"
ug4_house <- extract_electric(ug4_house,'HHID',path_elec, 'h10q1', 'h10q9')

# need to add different id variable to ug5_house
ug5_house %<>% left_join(ug5_a %>% select(HHID,hh), by = c('hhid' = 'hh')) %>% 
  rename(w5_hhid = hhid, hhid = HHID)
path_elec <- "../../Data/lsms/Uganda/UGA_2015/gsec10_1.dta"
ug5_house <- extract_electric(ug5_house,'hhid',path_elec, 'h10q1', 'h10q9')

path_elec <- "../../Data/lsms/Uganda/UGA_2018/HH/GSEC10_1.dta"
ug6_house <- extract_electric(ug6_house,'hhid',path_elec, 's10q01', 's10q09')

path_elec <- "../../Data/lsms/Uganda/UGA_2019/HH/gsec10_1.dta"
ug7_house <- extract_electric(ug7_house,'hhid',path_elec, 's10q01', 's10q09')

rm(path_elec)
#...............................................................................
##### Recode houseing variables #####
#...............................................................................

substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

recode_files <- list.files("../../Data/lsms/Uganda/recode/",full.names = T)
house_list <- list(ug1_house, ug2_house, ug3_house, ug4_house,ug5_house,ug6_house,ug7_house)

for(file in recode_files){
  file_name <- tail(strsplit(file,"/")[[1]],1)
  file_name <- sub(".csv","",file_name)
  var_name <- unlist(strsplit(file_name,"_recode"))[1]
  recode_df <- read.csv(file)
  is_w6_w7_file <- grepl("w6_w7",file_name)
  if(is_w6_w7_file){
    house_list[[6]] <- house_list[[6]] %>% 
      left_join(recode_df, by = var_name, all.x =T)
    house_list[[7]] <- house_list[[7]] %>% 
      left_join(recode_df, by = var_name, all.x =T)
    next
  }
  for(i in 1:7){
    if(var_name == 'toilet' & i %in% c(6,7)){
      next
    }
    else{
      house_list[[i]] <- house_list[[i]] %>% 
        left_join(recode_df, by = var_name, all.x =T)
    }
  }
}

ug1_house <- house_list[[1]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ug2_house <- house_list[[2]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ug3_house <- house_list[[3]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ug4_house <- house_list[[4]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ug5_house <- house_list[[5]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet,-hhid)
ug6_house <- house_list[[6]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)
ug7_house <- house_list[[7]] %>% select(-wall,-roof,-floor,-cooking_fuel,-watsup,-toilet)

rm(recode_files, house_list, is_w6_w7_file, file,file_name,i,recode_df)

#...............................................................................
##### Assets #####
#...............................................................................
assets = cbind.data.frame(code = c(7,6,10,11,12,16),
                          label = c("radio",'tv','bike','motorcycle','car','phone'))

early_ass <- list(ug1_ass,ug2_ass,ug3_ass,ug4_ass,ug5_ass,ug6_ass,ug7_ass)
names(early_ass) <- c('ug1_ass','ug2_ass','ug3_ass','ug4_ass','ug5_ass','ug6_ass','ug7_ass')
for(i in 1:length(early_ass)){
  if(i %in% c(6,7)){
    early_ass[[i]]$h14q2 = early_ass[[i]]$h14q02
    early_ass[[i]]$h14q3 = early_ass[[i]]$h14q03
  }
  early_ass[[i]] %<>% filter(h14q2 %in% assets$code) %>% 
    left_join(assets, by = c('h14q2' = 'code')) %>% 
    mutate(own = ifelse(h14q3 == 1,1,0)) %>% 
    select(matches('hhid'),label,own) %>% 
    pivot_wider(names_from = label, values_from = own)
}
list2env(early_ass,globalenv())
rm(early_ass)

# ug5_ass misses one id variable
ug5_ass %<>% left_join(ug5_a %>% select(HHID, hh), by = c('hhid' = 'HHID')) %>% 
  select(-hhid) %>% rename(w5_hhid = hh)

#...............................................................................
##### Consumption data #####
#.............................................................................
uga_cpi <- read.csv("../../Data/lsms/Uganda/uga_cpi.csv") %>% 
  filter(Country.Code == "UGA") %>% 
  select(-Series.Name, -Series.Code, -Country.Name)
names(uga_cpi) <- c("country",paste0("y_",2005:2019))
uga_cpi %<>% 
  pivot_longer(cols = starts_with("y_"), names_to = "year", names_prefix = "y_", values_to = "yearly_cpi") %>% 
  mutate(deflator_2017 = 166.7788/yearly_cpi, 
         year = as.numeric(year)) %>% 
  select(year, deflator_2017)
uga_2005_deflator = uga_cpi$deflator_2017[uga_cpi$year == 2005]
uga_2010_deflator = uga_cpi$deflator_2017[uga_cpi$year == 2010]


# for wave 3, the number of household members is not in the consumption data -> get from roster
ug3_roster <- read_dta("../../Data/lsms/Uganda/UGA_2011/GSEC2.dta") %>% 
  filter(h2q7 %in% c(1,2)) %>% # only use usual members present/absent just as in other waves
  select(HHID) %>% 
  group_by(HHID) %>%
  summarise(hsize = n()) %>% ungroup() %>% mutate_all(as.vector)

ug3_cons %<>% left_join(ug3_roster, by = 'HHID') 
rm(ug3_roster)

cons_list <- list(ug1_cons,ug2_cons,ug3_cons,ug4_cons,ug5_cons,ug6_cons,ug7_cons)
names(cons_list) <- c('ug1_cons','ug2_cons','ug3_cons','ug4_cons','ug5_cons','ug6_cons','ug7_cons')

for(i in 1:length(cons_list)){
  cons_list[[i]] <- cons_list[[i]] %>%  select(matches('hh|hhid|HHID'),
                                               matches('equiv_m|equiv'), 
                                               matches('hsize_m|hsize'),cpexp30) 
  if(i == 4){cons_list[[i]] %<>% select(-HHID_old)}
  names(cons_list[[i]]) <- c("hhid",'adulteq', 'hh_size', 'cpex30')
  cons_list[[i]]$hhid <- as.character(cons_list[[i]]$hhid)
  if(i > 4){
    cons_list[[i]]$cons_lcu_2017 = (cons_list[[i]]$cpex30 * uga_2010_deflator)/30 # divide by 30, cuz data is monthly
  }
  else{
    cons_list[[i]]$cons_lcu_2017 = (cons_list[[i]]$cpex30 * uga_2005_deflator)/30
  }
  cons_list[[i]]$cons_lcu_pc_2017 = cons_list[[i]]$cons_lcu_2017 / cons_list[[i]]$hh_size
  cons_list[[i]]$cons_usd_pc_2017 = cons_list[[i]]$cons_lcu_pc_2017 / 1221.08764648438
  cons_list[[i]] %<>% select(-cpex30,-cons_lcu_2017) 
}
list2env(cons_list,globalenv())
rm(cons_list)

#...............................................................................
##### merge data and split into attr and main data #####
#...............................................................................
ug1 <- ug1_cons %>% 
  left_join(ug1_house, by = c('hhid' = 'HHID')) %>% 
  left_join(ug1_ass, by = c('hhid' = 'HHID')) %>% 
  rename(w1_hhid = hhid) %>% 
  mutate(wave = 1, year = 2009)

ug2 <- ug2_cons %>% 
  left_join(ug2_house, by = c('hhid' = 'HHID')) %>% 
  left_join(ug2_ass, by = c('hhid' = 'HHID')) %>% 
  rename(w1_hhid = hhid)%>% 
  mutate(wave = 2, year = 2010)

ug3 <- ug3_cons %>% 
  left_join(ug3_house, by = c('hhid' = 'HHID')) %>% 
  left_join(ug3_ass, by = c('hhid' = 'HHID')) %>% 
  rename(w1_hhid = hhid)%>% 
  mutate(wave = 3, year = 2011)

ug4 <- ug4_cons %>% 
  left_join(ug4_house, by = c('hhid' = 'HHID')) %>% 
  left_join(ug4_ass, by = c('hhid' = 'HHID')) %>% 
  rename(w4_hhid = hhid) %>% 
  mutate(wave = 4, year = 2013) %>% 
  filter(w4_hhid %in% long_panel_ids$w4_hhid) %>% 
  left_join(long_panel_ids %>% select(w4_hhid,w1_hhid,cluster_id), by = 'w4_hhid') %>% 
  select(-w4_hhid) %>% relocate(w1_hhid, cluster_id)

ug5 <- ug5_cons %>% 
  rename(w5_hhid = hhid) %>% 
  left_join(ug5_house, by = 'w5_hhid') %>% 
  left_join(ug5_ass, by = 'w5_hhid') %>% 
  mutate(wave = 5, year = 2015) %>% 
  filter(w5_hhid %in% long_panel_ids$w5_hh) %>% 
  left_join(long_panel_ids %>% select(w5_hh,w1_hhid,cluster_id), by = c('w5_hhid' = 'w5_hh')) %>% 
  select(-w5_hhid) %>% relocate(w1_hhid, cluster_id)

ug6 <- ug6_cons %>% 
  left_join(ug6_house, by = 'hhid') %>% 
  left_join(ug6_ass, by = 'hhid') %>% 
  rename(w6_hhid = hhid) %>% 
  mutate(wave = 6, year = 2018) %>% 
  filter(w6_hhid %in% long_panel_ids$w6_hhid) %>% 
  left_join(long_panel_ids %>% select(w6_hhid,w1_hhid,cluster_id), by = 'w6_hhid') %>% 
  select(-w6_hhid) %>% relocate(w1_hhid, cluster_id)

ug7 <- ug7_cons %>% 
  left_join(ug7_house, by = 'hhid') %>% 
  left_join(ug7_ass, by = 'hhid') %>% 
  rename(w7_hhid = hhid) %>% 
  mutate(wave = 7, year = 2019) %>% 
  filter(w7_hhid %in% long_panel_ids$w7_hhid) %>% 
  left_join(long_panel_ids %>% select(w7_hhid,w1_hhid,cluster_id), by = 'w7_hhid') %>% 
  select(-w7_hhid) %>% relocate(w1_hhid, cluster_id)


#### create short panel dataset
short_panel <- rbind.data.frame(ug1,ug2,ug3) %>% 
  filter(w1_hhid %in% short_panel_ids$w1_hhid) %>% 
  left_join(short_panel_ids, by = 'w1_hhid') %>% 
  left_join(ug_geos, by = 'cluster_id') %>% 
  mutate(country = 'uga') %>% 
  mutate(cluster_id = paste(country,cluster_id,sep = "_")) %>% 
  rename(case_id = w1_hhid) %>% 
  relocate(country,year,wave,cluster_id,lat,lon,case_id) %>% 
  mutate_all(as.vector)

#### create long panel dataset
long_panel <- rbind.data.frame(ug1,ug2,ug3) %>% 
  filter(w1_hhid %in% long_panel_ids$w1_hhid) %>% 
  left_join(long_panel_ids %>% select(w1_hhid, cluster_id), by = 'w1_hhid') %>% 
  relocate(w1_hhid, cluster_id)

long_panel <- rbind.data.frame(long_panel,ug4,ug5,ug6,ug7) %>% 
  left_join(ug_geos, by = 'cluster_id') %>% 
  mutate(country = 'uga') %>% 
  mutate(cluster_id = paste(country,cluster_id,sep = "_")) %>% 
  rename(case_id = w1_hhid) %>% 
  relocate(country,year,wave,cluster_id,lat,lon,case_id) %>% 
  mutate_all(as.vector)

#### short panel attrition
# only consider households from the very first wave
short_panel_attr <- hhid_comm %>% 
  filter(cluster_id %in% short_eas) %>% 
  filter((w1_hhid %in% short_panel$case_id) == F) %>% 
  left_join(ug1, by = 'w1_hhid') %>% 
  left_join(ug_geos, by = 'cluster_id') %>% 
  mutate(country = 'uga') %>% 
  mutate(cluster_id = paste(country,cluster_id,sep = "_")) %>% 
  rename(case_id = w1_hhid) %>% 
  relocate(country,year,wave,cluster_id,lat,lon,case_id) %>% 
  mutate_all(as.vector)

long_panel_attr <- hhid_comm %>% 
  filter(cluster_id %in% long_eas) %>% 
  filter((w1_hhid %in% long_panel$case_id) == F) %>% 
  left_join(ug1, by = 'w1_hhid') %>% 
  left_join(ug_geos, by = 'cluster_id') %>% 
  mutate(country = 'uga') %>% 
  mutate(cluster_id = paste(country,cluster_id,sep = "_")) %>% 
  rename(case_id = w1_hhid) %>% 
  relocate(country,year,wave,cluster_id,lat,lon,case_id) %>% 
  mutate_all(as.vector)

#...............................................................................
##### save data #####
#...............................................................................
write.csv(short_panel,"../../Data/processed/uga_short_labels.csv")
write.csv(short_panel_attr,"../../Data/processed/uga_short_labels_attr.csv")
write.csv(long_panel,"../../Data/processed/uga_long_labels.csv")
write.csv(long_panel_attr,"../../Data/processed/uga_long_labels_attr.csv")





