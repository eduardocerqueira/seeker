#date: 2023-09-28T16:55:23Z
#url: https://api.github.com/gists/05d0c39c206dd8a91f90262eff993867
#owner: https://api.github.com/users/rursul

// for unix
ALTER TABLE `sheer_leads` ADD `timestamp` INT( 11 ) NULL;


ALTER TABLE `sheer_leads` ADD `month` varchar(12) NOT NULL


// удалить
ALTER TABLE sheer_leads
DROP COLUMN month
