library(tidyverse)
library(ggplot2)
library(GGally)
library(MASS)
library(gridExtra)
library(dplyr)
library(leaps)
library(rARPACK)
library(choroplethr)
library(choroplethrAdmin1)
library(choroplethrMaps)
library(randomForest)
require(glmnet)
require(C50)
require(gmodels)
require(xgboost)
require(boot)
require(msgl)
require(h2o)
require(RWeka)
require(RSNNS)
setwd("D:/大四下/data science/internship/house")
macro<-read_csv("macro.csv")
train<-read_csv("train.csv")
test<-read_csv("test.csv")
macro1<-macro%>%separate(timestamp,into=c("year","month","day"),sep="-")%>%unite(date,year,month,sep="-")%>%group_by(date)%>%
  summarise(GDP=mean(gdp_quart_growth),CPI=mean(cpi),MICEX=mean(micex),RATE=mean(deposits_rate))
macro1$CPIRATE<-lag((macro1$CPI-lag(macro1$CPI))/lag(macro1$CPI)*100)
macro1$MICEX<-lag((macro1$MICEX-lag(macro1$MICEX))/lag(macro1$MICEX)*100)
macro1$RATE<-lag(macro1$RATE-lag(macro1$RATE))

train1<-train%>%separate(timestamp,into=c("year","month","day"),sep="-")%>%unite(date,year,month,sep="-")
train1$product_type<-as.factor(train1$product_type)
train1$product_type<-as.numeric(train1$product_type=="Investment")
summary(train1$product_type)
train2<-train1%>%group_by(date)%>%summarise(PRODUCT=mean(product_type))

d1<-train2%>%left_join(macro1)
ggpairs(d1[,2:7])
qplot(date,PRODUCT,data=train2)
summary(train2$PRODUCT)


#product type------------------------------------------------------------------------------------------------------
test1<-test%>%separate(timestamp,into=c("year","month","day"),sep="-")%>%unite(date,year,month,sep="-")
test1$product_type<-as.factor(test1$product_type)
test1$product_type<-as.numeric(test1$product_type=="Investment")
summary(test1$product_type)
test1<-test1[,c(2,13)]%>%na.omit()
summary(test1$product_type)
test2<-test1%>%group_by(date)%>%summarise(PRODUCT=mean(product_type))
train2$x<-1
test2$x<-2
pro<-rbind(train2,test2)
qplot(date,PRODUCT,data=pro,col=x)
fft1<-fft(pro$PRODUCT)
fft1[abs(fft1)<1]=0
fft2<-fft(fft1,inverse=TRUE)/length(fft1)
fft2<-Re(fft2)
plot(fft2)
plot(pro$PRODUCT)
pro$fft<-fft2
pro$date<-ts(pro$date)
pro$index<-c(1:58)
lm1<-lm(PRODUCT~poly(index,4),data=pro)
lm2<-lm(PRODUCT~poly(index,6),data=pro)
lm3<-lm(PRODUCT~poly(index,8),data=pro)
layout(1)
plot(pro$PRODUCT)
lines(pro$index,lm1$fitted.values,col="red")
lines(pro$index,lm2$fitted.values,col="blue")
lines(pro$index,lm3$fitted.values,col="green")
p0<-plot(pro$index,pro$fft,col=pro$x,main="The demand of house",xlab = "Time from 2011 to 2016",ylab="Buy house for investment",geom="smooth")
p0<-ggplot(pro,aes(index,fft))
p0<-p0+geom_point(col=pro$x)+geom_smooth(aes(index,fft),method="lm",formula=y~poly(x,8))+labs(x="2011 to 2016",y="investment %",title="House demand")+
  theme(legend.position="topright")
lm4<-lm(fft~poly(index,4),data=pro)
lm5<-lm(fft~poly(index,6),data=pro)
lm6<-lm(fft~poly(index,8),data=pro)
lines(pro$index,lm4$fitted.values,col="red")
lines(pro$index,lm5$fitted.values,col="blue")
lines(pro$index,lm6$fitted.values,col="green")
legend("topright",legend = c("train data","test data","fitted model 1","fitted model 2"),col=c("black","red","blue","green"),
       lty=c(1,1,1,1))


#quantile price----------------------------------------------------------------------------------------------------------
train1$price<-train1$price_doc/train1$full_sq
summary(train1$price_doc)
summary(train1$full_sq)
summary(train1$price)
train1%>%filter(full_sq<10)
quantile(train1$full_sq,seq(0,1,0.05))
train1%>%filter(full_sq>200)
train1%>%filter(full_sq<20)
train3<-train1%>%filter(full_sq<=100&full_sq>=30)
train3<-train3%>%left_join(macro1)
train3$price<-train3$price_doc/train3$CPI*531/train3$full_sq
summary(train3$price)
train3<-train3[train3$price<2000000,]
qplot(train3$price)
qplot(log(train3$price))
train3$date<-ts(train3$date)
a <- data.frame(Time=c(time(train3$date)),Price=c(train3$price))
p <- ggplot(a,aes(x=Time,y=Price))
p+geom_point()
p1 <- ggplot(a,aes(x=Time,y=log(Price)))
p1+geom_point()
p1+geom_jitter()
train5<-train3%>%group_by(date)%>%summarise(q0.05=quantile(log(price),probs=seq(0,1,0.05))[2],q0.15=quantile(log(price),probs=seq(0,1,0.05))[4],
                                    q0.3=quantile(log(price),probs=seq(0,1,0.05))[7],q0.5=quantile(log(price),probs=seq(0,1,0.05))[11],
                                    q0.7=quantile(log(price),probs=seq(0,1,0.05))[15],q0.85=quantile(log(price),probs=seq(0,1,0.05))[18],
                                    q0.95=quantile(log(price),probs=seq(0,1,0.05))[20])
dim(train5)

p1<-qplot(data=train5,x=1:47,y=exp(q0.05),geom = "smooth")
p2<-qplot(data=train5,x=1:47,y=exp(q0.15),geom = "smooth")
p3<-qplot(data=train5,x=1:47,y=exp(q0.3),geom = "smooth")
p4<-qplot(data=train5,x=1:47,y=exp(q0.5),geom = "smooth")
p5<-qplot(data=train5,x=1:47,y=exp(q0.7),geom = "smooth")
p6<-qplot(data=train5,x=1:47,y=exp(q0.85),geom = "smooth")
p7<-qplot(data=train5,x=1:47,y=exp(q0.95),geom = "smooth")
grid.arrange(p1, p2, p3,p4,p5,p6,p7, ncol=3)
p1<-qplot(data=train5,x=1:47,y=exp(q0.05))
p2<-qplot(data=train5,x=1:47,y=exp(q0.15),main = "Cheap house price",xlab="2011 to 2015",ylab="house price")
p3<-qplot(data=train5,x=1:47,y=exp(q0.3))
p4<-qplot(data=train5,x=1:47,y=exp(q0.5),main = "Normal house price",xlab="2011 to 2015",ylab="house price")
p5<-qplot(data=train5,x=1:47,y=exp(q0.7))
p6<-qplot(data=train5,x=1:47,y=exp(q0.85),main = "Expensive house price",xlab="2011 to 2015",ylab="house price")
p7<-qplot(data=train5,x=1:47,y=exp(q0.95))
grid.arrange( p2, p4,p6,p0, ncol=2)


#state--------------------------------------------------------------------------------------------------------------------
train4<-train%>%filter(state!="NA")
train4$state<-as.numeric(train4$state)
train$state<-as.numeric(train$state)
table(train4$state)
train%>%filter(state=="NA")


#location----------------------------------------------------------------------------------------------------------------------
train1$sub_area<-as.factor(train1$sub_area)
summary(train1$sub_area)
str(train1$sub_area)
train3$sub_area<-as.factor(train3$sub_area)
summary(train3$sub_area)
train6<-train3%>%group_by(sub_area)%>%summarise(price2=mean(price))
bp<-ggplot(train3,aes(sub_area,price))
bp+geom_boxplot()


#floor material year-------------------------------------------------------------------------------------------------------------------
summary(train1$floor)
train3$floor<-as.factor(train3$floor)
train3$material<-as.factor(train3$material)
train3$floor<-as.factor(train3$floor)
bp1<-ggplot(train3,aes(floor,price))
bp1+geom_boxplot()
bp2<-ggplot(train3,aes(material,price))
bp2+geom_boxplot()
train3$build_year<-as.factor(train3$build_year)
bp3<-ggplot(train3,aes(build_year,price))
bp3+geom_boxplot()


#variable-------------------------------------------------------------------------------------------------------------------
train7<-train3%>%dplyr::select(id,date,day,full_sq,raion_popul,green_zone_part,indust_part,school_education_centers_top_20_raion,
                healthcare_centers_raion,university_top_20_raion,sport_objects_raion,culture_objects_top_25_raion,
                shopping_centers_raion,big_market_raion,nuclear_reactor_raion,detention_facility_raion,cafe_avg_price_500,
                price,CPI,RATE)
train7$big_market_raion<-as.factor(train7$big_market_raion)%>%as.numeric()-1
train7$nuclear_reactor_raion<-as.factor(train7$nuclear_reactor_raion)%>%as.numeric()-1
train7$detention_facility_raion<-as.factor(train7$detention_facility_raion)%>%as.numeric()-1
table(train7$detention_facility_raion)
table(train7$nuclear_reactor_raion)
table(train7$big_market_raion)
price_na_cafe<-train7[is.na(train7$cafe_avg_price_500),]%>%dplyr::select(price)
qplot(price,data=price_na_cafe)
train8<-train7[is.na(train7$cafe_avg_price_500),]
train9<-train7[!is.na(train7$cafe_avg_price_500),]
train10<-train3%>%dplyr::select(id,date,day,full_sq,raion_popul,green_zone_part,indust_part,school_education_centers_top_20_raion,
                                healthcare_centers_raion,university_top_20_raion,sport_objects_raion,culture_objects_top_25_raion,
                                shopping_centers_raion,big_market_raion,nuclear_reactor_raion,detention_facility_raion,cafe_avg_price_2000,
                                price,CPI,RATE)
train10<-train10[!is.na(train10$cafe_avg_price_2000),]
dim(train10)
train10$big_market_raion<-as.factor(train10$big_market_raion)%>%as.numeric()-1
train10$nuclear_reactor_raion<-as.factor(train10$nuclear_reactor_raion)%>%as.numeric()-1
train10$detention_facility_raion<-as.factor(train10$detention_facility_raion)%>%as.numeric()-1
lm0<-lm(price~.,data=train10[,4:18])
summary(lm0)
plot(lm0)
#heavy both size
lm1<-lm(log(price)~.,data=train10[,4:18])
summary(lm1)
plot(lm1)
#left heavy tail
step(lm0)
step(lm0,k=log(dim(train10)[1]))

#remove investment effect--------------------------------------------------------------------------------------------------
names(pro)[5]<-"timeindex"
pro$fit<-lm6$fitted.values
train11<-train10%>%left_join(pro[,c(1,6)])
train11$price0<-train11$price/train11$fit*0.5177069
hist(train11$price0)
rmlm0<-lm(price0~.,data=train11[,c(4:17,22)])
summary(rmlm0)
step(rmlm0)
step(rmlm0,k=log(dim(train11)[1]))

train12<-train11%>%group_by(date)%>%summarise(q0.05=quantile(log(price0),probs=seq(0,1,0.05))[2],q0.15=quantile(log(price0),probs=seq(0,1,0.05))[4],
                                            q0.3=quantile(log(price0),probs=seq(0,1,0.05))[7],q0.5=quantile(log(price0),probs=seq(0,1,0.05))[11],
                                            q0.7=quantile(log(price0),probs=seq(0,1,0.05))[15],q0.85=quantile(log(price0),probs=seq(0,1,0.05))[18],
                                            q0.95=quantile(log(price0),probs=seq(0,1,0.05))[20])
dim(train12)
p1<-qplot(data=train12,x=1:47,y=exp(q0.05))
p2<-qplot(data=train12,x=1:47,y=exp(q0.15))
p3<-qplot(data=train12,x=1:47,y=exp(q0.3))
p4<-qplot(data=train12,x=1:47,y=exp(q0.5))
p5<-qplot(data=train12,x=1:47,y=exp(q0.7))
p6<-qplot(data=train12,x=1:47,y=exp(q0.85))
p7<-qplot(data=train12,x=1:47,y=exp(q0.95))
grid.arrange(p1, p2, p3,p4,p5,p6,p7, ncol=3)
rmlm1<-lm(price~.,data=train11[,c(4:18,21)])
summary(rmlm1)
train11<-train11[,-5]

#kmeans------------------------------------------------------------------------------------------------------------------------
clusters<-kmeans(scale(train11[,4:17]),centers=7,iter.max = 10000,nstart=10000)
clusters$cluster
clusters$size
train11$group<-clusters$cluster%>%as.factor()

kp<-ggplot(data=train11,aes(group,price))
kp+geom_col()
kp+geom_boxplot()

#-------------------------------------------------------------------------------------------------------------------------------
plot(rmlm1)
train11$date<-as.factor(train11$date)
bp4<-ggplot(train11,aes(date,log(price0)))
bp4+geom_boxplot()
bp5<-ggplot(train11,aes(date,price0))
bp5+geom_boxplot()
train11$low=0
train11$mid=0
train11$high=0
train11$low[log(train11$price0)<=11]=1
train11$high[train11$price0>=250000]=1
train11$mid[train11$low==0&train11$high==0]=1
mid<-train11%>%filter(mid==1)
high<-train11%>%filter(high==1)
low<-train11%>%filter(low==1)
plot(density(low$cafe_avg_price_2000),col="red")
lines(density(mid$cafe_avg_price_2000),col="blue")
lines(density(high$cafe_avg_price_2000),col="green")

#area-----------------
area<-train3%>%dplyr::select(id,date,day,full_sq,sub_area)
train12<-train11%>%left_join(area)
bp6<-ggplot(train12,aes(sub_area,log(price0)))
bp6+geom_boxplot()

# R to python-----------------
write_csv(train11[,22:24],"3class.csv")
write_csv(train11[,4:16],"X.csv")
write_csv(train11[,17:21],"Y.csv")

# model--------------------------------------------------------------------------------------------------------------
X<-train11[train11$mid==1,]
dim(X)
sam<-sample(23768,floor(23768*0.3))
X_train<-train11[-sam,4:16]
X_test<-train11[sam,4:16]
y_train<-log(train11[-sam,21])
y_test<-log(train11[sam,21])
m1<-randomForest(price0~.,data=cbind(X_train,y_train),mtry = 8)
summary(m1)
pre1<-predict(m1,X_test)
mean(abs(pre1-as.matrix(y_test)))#0.3365

m2<-M5P(price0~.,data=cbind(X_train,y_train))
summary(m2)
pre2<-predict(m2,X_test)
mean(abs(pre2-as.matrix(y_test)))#0.3339
mae=list(0)
for(i in 1:100)
{
m3[[i]]<-cv.glmnet(as.matrix(X_train),as.matrix(y_train),family="gaussian",nfolds = 5,alpha=i/100)
m4[[i]]<-glmnet(as.matrix(X_train),as.matrix(y_train),family="gaussian",lambda = m3[[i]]$lambda.min)
mae[[i]]<-mean(abs(predict.glmnet(m4[[i]], as.matrix(X_test),s = m4[[i]]$lambda)-as.matrix(y_test)))
}
0.3462
m5<-lm(price0~.,data=cbind(X_train,y_train))
pre5<-predict(m5,X_test)
mean(abs(pre5-as.matrix(y_test)))
#  NN 0.3683 python keras-----------------------------------------------------------------------

#XGBoost-------------------------------------------------------------------------------------
train_x<-as.matrix(X_train)
train_y<-as.matrix(y_train)
test_x<-as.matrix(X_test)
test_y<-as.matrix(y_test)
dtrain <- xgb.DMatrix(train_x, label = train_y)
dtest <- xgb.DMatrix(test_x, label = test_y) 
watchlist <- list(eval = dtest, train = dtrain)
param <- list(subsample=0.7,eta=0.5,gamma=0.5,max_depth = 4, silent = 1, objective = "reg:linear",eval_metric = "mae")
bst <- xgb.train(param, dtrain, nrounds = 100, watchlist)#0.3036

param <- list(subsample=0.7,eta=0.5,gamma=0.5,max_depth = 4, silent = 1, objective = "reg:linear",eval_metric = "mae")
bst <- xgb.train(booster="gblinear",param, dtrain, nrounds = 100, watchlist)#0.4191

param <- list(subsample=0.7,eta=0.5,gamma=0.5,max_depth = 4, silent = 1, objective = "reg:linear",eval_metric = "mae")
bst <- xgb.train(booster="dart",param, dtrain, nrounds = 100, watchlist)#0.3065

bst1<-xgb.importance(colnames(train_x),bst)
xgb.plot.importance(bst1,rel_to_first =FALSE)
