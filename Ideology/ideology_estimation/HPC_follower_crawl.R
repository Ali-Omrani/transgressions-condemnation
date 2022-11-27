#toInstall <- c("ggplot2", "scales", "R2WinBUGS", "devtools", "yaml", "httr", "RJSONIO")
#install.packages(toInstall, repos = "http://cran.r-project.org")
#library(devtools)
#install_github("pablobarbera/twitter_ideology/pkg/tweetscores")

library(tweetscores)

oath_file = "../../../data/auth/twitter_app_aouth - Sheet1.csv"


args <- commandArgs(trailingOnly = TRUE)
print(args[1])
user_ids <-read.csv(args[1])

accounts.left <- user_ids$user_id
while (length(accounts.left) > 0){
  cat("users left")
  cat(length(accounts.left))
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