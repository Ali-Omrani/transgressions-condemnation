twitter_file = "../../data/filtered_condemnation_RTs.csv"
user_id_ideology_file = "../../data/user_id_political_ideologies.rdata"
RT_user_id_ideology_file = "../../data/RT_user_id_political_ideologies.rdata"

library(ggplot2)
library(MASS)
library(dplyr)
library(glmmTMB)
library(AER)
library(pscl)
# -----------------------------loading data------------------------------
twitter_df = read.csv(twitter_file)

load(RT_user_id_ideology_file)
RT_user_ideologies <- res
load(user_id_ideology_file)
og_user_ideologies <- res

user_ideology_estimates = og_user_ideologies$rowcoord[,1]
user_ids = og_user_ideologies$rownames
RT_user_ideology_estimates = RT_user_ideologies$rowcoord[,1]
RT_user_ids = RT_user_ideologies$rownames

user_ideology_df = data.frame(user_ideology_estimates, user_ids)
RT_user_ideology_df = data.frame(RT_user_ideology_estimates, RT_user_ids)


Y = twitter_df$favoritesCount
X = twitter_df$retweetCount

summary(Y)
summary(X)
#----------------------------poisson----------------------
poisson <- glm(Y~X, family= poisson)
summary(poisson)


#--------------------------Dispersion test-------------------
# from AER package

dispersiontest(poisson)
dispersiontest(poisson, trafo = 2)

#---------------------------negative binomial---------------
negbin <- glm.nb(Y ~ X)
summary(negbin)


#----------------------------hurdle (truncated poisson) model---------------
hpossion <- hurdle(Y~X | X, link="logit", dist="poisson") # second x affect 0 or positive decision
summary(hpossion)


#----------------------------hurdle (truncated negative binomial) model---------------
hpossion <- hurdle(Y~X | X, link="logit", dist="negbin") # second x affect 0 or positive decision
summary(hpossion)


# ----------------------------zero inflated =poisson -----------------------------------
zip <- zeroinfl(Y~X | X, link="logit", dist="poisson")
summary(zip)

#----------------------------zero inflated negative binomial----------------
zinb <- zeroinfl(Y~X | X, link="logit", dist="poisson")
summary(zinb)



fav_sev_glm_nb = glmmTMB(favoritesCount ~ severity_prediction, data = twitter_df, family = nbinom1, ziformula = ~1)
summary(fav_sev_glm_nb)
fav_sev_nb = glm.nb(favoritesCount ~ severity_prediction, data = twitter_df)
#fav_sev = lm(favoritesCount ~ severity_prediction, data = twitter_df)
summary(fav_sev_nb)

ggplot(twitter_df,aes(severity_prediction, favoritesCount)) +
  stat_summary(fun.data=mean_cl_normal) + 
  geom_smooth(method='lm', formula= y~x) +
  geom_point(color='blue')


rt_sev = lm(retweetCount ~ severity_prediction, data = twitter_df)
rt_sev_nb = glm.nb(retweetCount ~ severity_prediction, data = twitter_df)

summary(rt_sev_nb)

