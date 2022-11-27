library(tweetscores)
data_dir <- here::here("poli_account_followers/")
outfolder <- 'followers-lists-202008/'
outfolderPath <- file.path(data_dir, outfolder)
dir.create(outfolderPath, showWarnings = FALSE)
polsfile <- "accounts-twitter-data-2020-08.csv"
oauth_folder <- here::here("poli_account_followers/twitter_app_aouth - Sheet1.csv")

#load poli_estimate_df
load(file="./poli_estimate_ours.Rda")

sampled_poli_estimate_df = poli_estimate_df[sample(nrow(poli_estimate_df),1000),]
sampled_poli_estimate_df['barbera_estimate'] <- NA
#while(sum(is.na(sampled_poli_estimate_df$barbera_estimate))>0){
for (row in 1:nrow(sampled_poli_estimate_df)) {
  
  if (!is.na(sampled_poli_estimate_df[row, "barbera_estimate"]) ) {
    print("skipping")
    print(row)
    next
  }
  
  tryCatch(
    expr = {
        print(row)
        curr_user_id = sampled_poli_estimate_df[row,"user_id"]
        #print(sampled_poli_estimate_df[row,])
        print("before friends")
        friends = getFriends(user_id = curr_user_id,oauth = oauth_folder, verbose = FALSE)
        print("got friends")
        barbera_estimate = estimateIdeology2(user = curr_user_id, friends = friends)
        sampled_poli_estimate_df[row, "barbera_estimate"] = barbera_estimate
  
      
    },
    error = function(e){ 
      # (Optional)
      print("error")
      
      # Do this if an error is caught...
    }
  )
}
#}

two_estimates_df = sampled_poli_estimate_df
save(two_estimates_df, file="corr_estimate_df.Rda")

two_estimates_df
library(dplyr)
cleaned_df %>% dplyr::drop_na(two_estimates_df)


cleaned_df = two_estimates_df[complete.cases(two_estimates_df),]
cleaned_df$bin_estimate_ours <- ifelse(cleaned_df$estimate_ours > 0, 1, -1)
cleaned_df$bin_barbera_estimate <- ifelse(cleaned_df$barbera_estimate < 0, 1, -1)
sum(cleaned_df$bin_barbera_estimate == cleaned_df$bin_estimate_ours)
