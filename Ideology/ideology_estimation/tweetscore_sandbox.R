toInstall <- c("ggplot2", "scales", "R2WinBUGS", "devtools", "yaml", "httr", "RJSONIO")
install.packages(toInstall, repos = "http://cran.r-project.org")
library(devtools)
install_github("pablobarbera/twitter_ideology/pkg/tweetscores")
library(tweetscores)
pred_data = read.csv("../../../data/5_mil_7days_metoo.csv")
user_ids = read.csv("../../../data/user_ids_5mil_7days.csv")
unique_user_ids = unique(user_ids$X_id)
subset_unique_ids = head(unique_user_ids,100)
oath_file = "../../../data/auth/twitter_app_aouth - Sheet1.csv"
result_df <-data.frame("user_id","ideology")
for (u_id in subset_unique_ids){
  friends <- getFriends(user_id=u_id, oauth = oath_file)
  result <- estimateIdeology2(u_id, friends = friends)
#  result_df[nrow(result_df) + 1,] = c(u_id,result)
  
}
user <- subset_unique_ids[1]
friends <- getFriends(user_id=subset_unique_ids, oauth = oath_file)
result <- estimateIdeology2(user, friends = friends)

user_info <-getUsersBatch(ids = user, oauth = oath_file)



#--------------------------------Loop with Exception---------------------------------------
accounts.left <- unique_user_ids
while (length(accounts.left) > 0){
  
  # sample randomly one account to get followers
  new.user <- sample(accounts.left, 1)
  #new.user <- accounts.left[1]
  #cat(new.user, "---", users$followers_count[users$screen_name==new.user], 
      #" followers --- ", length(accounts.left), " accounts left!\n")    
  
  # download followers (with some exception handling...) 
  error <- tryCatch(friends <- getFriends(user_id=new.user, oauth = oath_file), error=function(e) e)
  #error <- tryCatch(followers <- getFollowers(screen_name=new.user,
  #                                            oauth=oauth_folder, sleep=3, verbose=FALSE), error=function(e) e)
  if (inherits(error, 'error')) {
    cat("Error! On to the next one...")
    next
  }
  
  # save to file and remove from lists of "accounts.left"
  file.name <- paste0("../../../data/followers/", new.user, ".rdata")
  save(friends, file=file.name)
  accounts.left <- accounts.left[-which(accounts.left %in% new.user)]
  
}
