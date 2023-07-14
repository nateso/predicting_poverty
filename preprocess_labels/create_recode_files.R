#### make recode files

#### Ethiopia ####
rm(list = ls())

watsup_recode_w3 <- data.frame(
  watsup = c(1:15),
  watsup_qual = c(5,5,4,4,4,3,4,3,1,2,1,2,2,1,3)
)

for(dat in ls()){
  path = paste0("~/Documents/AngSt/master_thesis/data_analysis/Data/lsms/Ethiopia/recode/",
                dat,".csv")
  df = get(dat)
  write.csv(df,path,row.names = F)
  remove(df, path)
} 

#### Malawi ####
rm(list = ls())

wall_recode <- data.frame(
  wall = c(1:9),
  wall_qual = c(1,1,2,3,5,5,3,4,3)
)

roof_recode <- data.frame(
  roof = c(1:6),
  roof_qual = c(1,4,5,5,3,3)
)

floor_recode <- data.frame(
  floor = c(1:6),
  floor_qual = c(1,2,4,4,5,3)
)

toilet_recode <- data.frame(
  toilet = c(1:6),
  toilet_qual = c(5,3,4,3,1,3)
)

watsup_recode <- data.frame(
  watsup = c(1:16),
  watsup_qual = c(5,5,3,3,2,4,3,4,2,1,1,1,1,2,1,3)
)
 
cooking_fuel_recode <- data.frame(
  cooking_fuel = c(1:10),
  cooking_fuel_qual = c(1,1,4,5,5,2,1,2,1,3)
)

# same for w2, w3,  

watsup_recode_w4 <- data.frame(
  watsup = c(1:18),
  watsup_qual = c(5,5,3,3,2,4,3,4,2,1,1,1,1,2,1,3,4,2)
)

toilet_recode_w4 <- data.frame(
  toilet = c(1:13),
  toilet_qual = c(5,5,5,5,5,3,4,3,1,3,2,1,3)
)

for(dat in ls()){
  path = paste0("~/Documents/AngSt/master_thesis/data_analysis/Data/lsms/Malawi/recode/",
                dat,".csv")
  df = get(dat)
  write.csv(df,path,row.names = F)
  remove(df, path)
} 



#### Nigeria ####
rm(list = ls())
wall_recode_w1 <- data.frame(
  wall = c(1:9),
  wall_qual = c(1,1,2,3,5,5,3,4,3)
)

wall_recode_w2 <- data.frame(
  wall = c(0:11),
  wall_qual = c(1,1,1,2,3,5,5,3,4,5,3,3)
) 

wall_recode_w3 <- data.frame(
  wall = c(0:11),
  wall_qual = c(1,1,1,2,3,5,5,3,4,5,3,3)
)

wall_recode_w4 <- data.frame(
  wall = c(1:9),
  wall_qual = c(1,3,3,4,5,3,4,2,3)
)

roof_recode_w1 <- data.frame(
  roof = c(1:7), 
  roof_qual = c(1,4,5,5,3,4,3)
)

roof_recode_w2 <- data.frame(
  roof = c(1:7), 
  roof_qual = c(1,4,5,5,3,4,3)
)

roof_recode_w3 <- data.frame(
  roof = c(1:7), 
  roof_qual = c(1,4,5,5,3,4,3)
)

roof_recode_w4 <- data.frame(
  roof = c(1:10),
  roof_qual = c(1,4,5,5,3,4,1,5,4,3)
)

floor_recode_w1 <- data.frame(
  floor = c(1:7),
  floor_qual = c(1,2,4,4,5,3,5)
)

floor_recode_w2 <- data.frame(
  floor = c(1:7),
  floor_qual = c(1,2,4,4,5,3,5)
)

floor_recode_w3 <- data.frame(
  floor = c(1:7),
  floor_qual = c(1,2,4,4,5,3,5)
)

floor_recode_w4 <- data.frame(
  floor = c(1:7),
  floor_qual = c(1,2,4,4,5,3,5)
)

cooking_fuel_recode_w1 <- data.frame(
  cooking_fuel = c(1:10),
  cooking_fuel_qual = c(1,1,2,1,4,5,5,4,3,3)
)

cooking_fuel_recode_w2 <- data.frame(
  cooking_fuel = c(1:9),
  cooking_fuel_qual = c(1,1,2,1,4,5,5,5,3)
)

cooking_fuel_recode_w3 <- data.frame(
  cooking_fuel = c(1:9),
  cooking_fuel_qual = c(1,1,2,1,4,5,5,5,3)
)

cooking_fuel_recode_w4 <- data.frame(
  cooking_fuel = c(1:18),
  cooking_fuel_qual = c(4,2,2,1,4,1,1,2,3,3,4,4,5,5,5,5,1,3)
)

watsup_recode_w1 <- data.frame(
  watsup = c(1:10),
  watsup_qual = c(5,5,4,4,3,4,4,1,1,3)
) 

watsup_recode_w2 <- data.frame(
  watsup = c(1:12),
  watsup_qual = c(5,5,4,4,3,4,4,1,1,1,1,3)
) 

watsup_recode_w3 <- data.frame(
  watsup = c(1:12),
  watsup_qual = c(5,5,4,4,3,4,4,1,1,1,1,3)
)

watsup_recode_w4 <- data.frame(
  watsup = c(1:17),
  watsup_qual = c(5,5,2,2,4,4,3,4,3,1,1,1,4,1,1,1,3)
)

toilet_recode_w1 <- data.frame(
  toilet = c(1:9),
  toilet_qual = c(1,4,5,5,1,4,3,3,3)
)

toilet_recode_w2 <- data.frame(
  toilet = c(1:9),
  toilet_qual = c(1,4,5,5,1,4,3,3,3)
)

toilet_recode_w3 <- data.frame(
  toilet = c(1:9),
  toilet_qual = c(1,4,5,5,1,4,3,3,3)
)

toilet_recode_w4 <- data.frame(
  toilet = c(1:14),
  toilet_qual = c(5,5,5,5,5,3,4,3,3,1,2,1,3,5)
)


for(dat in ls()){
  path = paste0("~/Documents/AngSt/master_thesis/data_analysis/Data/lsms/Nigeria/recode/",
                dat,".csv")
  df = get(dat)
  write.csv(df,path,row.names = F)
  remove(df, path)
} 

#### Uganda ####
rm(list = ls())

roof_recode <- data.frame(
  roof = c(1:8,96,10:15),
  roof_qual = c(1,1,2,4,4,5,3,5,3,4,5,4,5,3,1)
)

wall_recode <- data.frame(
  wall = c(1:8,96,10:17),
  wall_qual = c(1,1,3,3,5,5,5,3,3,5,5,5,3,3,3,1,4)
)

floor_recode <- data.frame(
  floor = c(1:7,96,10:16),
  floor_qual = c(1,1,4,5,3,3,4,3,4,3,3,4,1,4,5)
)

watsup_recode <- data.frame(
  watsup = c(1:9,96,10:22),
  watsup_qual = c(5,4,4,4,3,1,2,4,1,3,5,5,4,4,3,4,3,1,2,2,4,1,1)
)

toilet_recode <- data.frame(
  toilet = c(1:9,96,10:17),
  toilet_qual = c(4,3,3,2,3,5,4,1,3,3,5,3,4,3,3,2,3,1)
)

cooking_fuel_recode <- data.frame(
  cooking_fuel = c(1:9,96),
  cooking_fuel_qual = c(5,5,4,1,2,2,1,1,3,3)
)

toilet_recode_w6_w7 <- data.frame(
  toilet = c(11:14,18,21:23,31,41,51,95,96),
  toilet_qual = c(5,5,5,5,5,3,4,3,3,1,2,1,3)
)

for(dat in ls()){
  path = paste0("~/Documents/AngSt/master_thesis/data_analysis/Data/lsms/Uganda/recode/",
                dat,".csv")
  df = get(dat)
  write.csv(df,path,row.names = F)
  remove(df, path)
} 



#### Tanzania ####
rm(list = ls())
wall_recode <- data.frame(
  wall = c(1:7),
  wall_qual = c(1,1,1,3,5,5,3)
)

roof_recode <- data.frame(
  roof = c(1:7),
  roof_qual = c(1,1,5,4,4,5,3)
)

floor_recode <- data.frame(
  floor = c(1:3),
  floor_qual = c(1,5,3)
)

watsup_recode_w1 <- data.frame(
  watsup = c(1:11),
  watsup_qual = c(5,5,4,4,2,2,4,3,1,1,3)
)

watsup_recode_w2_w3 <- data.frame(
  watsup = c(1:14),
  watsup_qual = c(5,5,4,4,2,2,2,4,3,3,2,1,1,3)
)

watsup_recode_w4_w5 <- data.frame(
  watsup = c(1:12),
  watsup_qual = c(5,4,4,3,4,3,1,1,2,2,1,3)
)

toilet_recode_w1 <- data.frame(
  toilet = c(1:5),
  toilet_qual = c(5,3,4,3,1)
)

toilet_recode_w2 <- data.frame(
  toilet = c(1:8),
  toilet_qual = c(1,5,4,3,4,3,4,3)
)

toilet_recode_w3_w4_w5 <- data.frame(
  toilet = c(1:9),
  toilet_qual = c(1,3,4,4,3,4,5,4,3)
)

cooking_fuel_recode <- data.frame(
  cooking_fuel = c(1:8),
  cooking_fuel_qual = c(1,4,5,5,2,1,5,3)
)

floor_recode_w5_2 <- data.frame(
  floor = c(1:6),
  floor_qual = c(1,4,3,5,4,3)
)


for(dat in ls()){
  path = paste0("~/Documents/AngSt/master_thesis/data_analysis/Data/lsms/Tanzania/recode/",
                dat,".csv")
  df = get(dat)
  write.csv(df,path,row.names = F)
  remove(df, path)
} 


#### Tanzania second round ####
rm(list = ls())










