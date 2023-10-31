#date: 2023-10-31T16:51:26Z
#url: https://api.github.com/gists/37e413e8245b46da03e4a31f60cc8527
#owner: https://api.github.com/users/remkus

#! /bin/bash

# Author: Birgit Olzem aka @CoachBirgit
# Version: 1.0
# Created on: January 19th 2022
# Requires WP-CLI, cURL, wget

# credits for the origin idea to Jeremy Herve: https://jeremy.hu/dev-environment-laravel-valet-wp-cli/

### How to use
# Use WP-CLI on Laravel/Valet to download, install and configure WordPress like a breeze. Generate some dummy content with attached featured images. 
# We call the Unsplash API via picsum.photos with the image specs w1920, h1200, random, grayscale.
# Create a local folder where you want to download the images from Unsplash and adjust the path.
# install and activate your go-to plugins and favorite theme while creating the site

### let the magic happen!

echo "What do you want your site to be called? No Spaces, just lowercase letters please."
read site_name

cd ~/Sites/dev/
mkdir $site_name
cd $site_name

echo "Launch a new Valet site first."
valet park
valet link

echo "Securing Site URL"
valet secure

# We will need a database for that WordPress site.
echo "Creating Database"
mysql -uroot -e "CREATE DATABASE $site_name"

echo "Now let's install WordPress"
wp core download
wp config create --dbname=$site_name --dbuser=root --dbpass=
# wp config set WP_DEBUG true

wp core install --url=https: "**********"

# choose your tagline
echo "What do want to display as tagline in your blog description?"
read blog_description
wp option update blogdescription $blog_description

# Remove example posts, pages, plugins and inactive themes
echo "Now let's do some cleaning"
wp theme delete $(wp theme list --status=inactive --field=name)
wp plugin delete --all
wp post delete $(wp post list --post_type='page' --format=ids) --force
wp post delete $(wp post list --post_type='post' --format=ids) --force
wp comment delete $(wp comment list --status=approved --format=ids)

# Rewrite permalink structure
echo "Now let's set permalink structure"
wp rewrite structure '/%postname%/' --hard
wp rewrite flush --hard

# Generate 10 posts with lorem ipsum content 
echo "Generate some posts"
curl http://loripsum.net/api/5 | wp post generate --post_content --count=10

# Download random images from Unsplash into a local folder, import them and set each as featured image related to the post
echo "Get POST_ID, download image and set it as featured image"
GET_POST_ID="$(wp post list --post_type=post --field=ID --format=csv)"

mkdir mediaimport
for post_id in ${GET_POST_ID[0]}; do
    wget https://picsum.photos/1920/1200/\?random\&grayscale -O ~/Sites/dev/$site_name/mediaimport/unsplash_$post_id.jpg;
    sleep 1m
    wp media import ~/Sites/dev/$site_name/mediaimport/unsplash_$post_id.jpg --post_id=$post_id --title="A dummy picture for $post_id" --featured_image
done

echo "we'll need some plugins"
wp plugin install elementor --activate

# You can install your premium plugins from a local folder and activate them directly
# wp plugin install ~/Sites/site-plugins/elementor-pro-3.5.2.zip --activate 

# You can also activate your Elementor Pro license on the fly

# echo "we'll activate the Elementor Pro license"
# wp elementor-pro license activate xoxoxoxoxoxoxoxoxoxoxo

echo "we'll install and activate the theme Kadence"
wp theme install kadence --activate

echo "Excellent. Your new site is ready!"
open http://$site_name.wp/wp-admin/

exit 0; theme Kadence"
wp theme install kadence --activate

echo "Excellent. Your new site is ready!"
open http://$site_name.wp/wp-admin/

exit 0;