#date: 2023-11-30T17:08:03Z
#url: https://api.github.com/gists/a68dfea7deb6d79f02ae2760156b9c62
#owner: https://api.github.com/users/ameerkat

##
# Shopify App Block Loader
# This script renders block liquid files to HTML, allowing you to load them
# up and view them locally in isolation. This allows you to load them up for
# testing and to see how they operate. For example we have a complex javascript
# that runs and we need to see how it interacts with a complete page. This is
# a rough approximation of what is necessary for shopify. Does not include out
# of block scope elements but does include the wrapping block div. Note this 
# does not render an entire product page. It only renders the block itself.
##
# Requirements
# pip install python-liquid
##
# What you need to configure
# * The product id if you use it
# * Your shopify store permanent domain
# * All directories (BLOCKS_DIRECTORY, SNIPPETS_DIRECTORY, ASSETS_DIRECTORY, OUTPUT_DIRECTORY)
##

from liquid import Template, Environment, FileSystemLoader
import os
import uuid
import logging
import re
import json

# Set this to your blocks directory, in our case our test script is located
# in Shopify/Alpha-App/test/extensions/dynamic_loader, where the test directory
# is adjacent to the blocks directory at the Alpha-App root level.
BLOCKS_DIRECTORY = "../../extensions/product-page-personalization/blocks"
SNIPPETS_DIRECTORY = "../../extensions/product-page-personalization/snippets"
# relative to build output directory
ASSETS_DIRECTORY = "../../../extensions/product-page-personalization/assets"

OUTPUT_DIRECTORY = "./generated"

def get_block_params(block_name, schema):
    block_id = str(uuid.uuid4())

    settings = {}
    for setting in schema["settings"]:
        if "default" in setting:
            settings[setting["id"]] = setting["default"]

    return {
        "block": {
            "id": block_id,
            "settings": settings
        },
        "product": {
            "id": 0000000000001
        },
        "shop": {
            "permanent_domain": "yourstore.myshopify.com"
        }
    }

def define_filters(env):
    def asset_url(asset_name):
        # one additional layer of depth since this is added to build folder
        # TODO This needs to be more flexible
        return f"{ASSETS_DIRECTORY}/{asset_name}"
    env.filters["asset_url"] = asset_url

    def stylesheet_tag(asset_url):
        return f"<link rel=\"stylesheet\" href=\"{asset_url}\">"
    env.filters["stylesheet_tag"] = stylesheet_tag

    def json_filter(data):
        return json.dumps(data)
    env.filters["json"] = json_filter

def extract_schema(template_string):
    start_schema = re.search('{%\s*schema\s*%}', template_string)
    if start_schema is None:
        raise Exception("No schema found for template.")
    end_schema = re.search('{%\s*endschema\s*%}', template_string)
    if end_schema is None:
        raise Exception("Syntax error when parsing schema, failed to locate end schema tag.")
    
    schema_string = template_string[start_schema.end():end_schema.start()]
    template_string = template_string[:start_schema.start()] + template_string[end_schema.end():]

    # Wrap template string in wrapping div
    template_string = "<div id='shopify-block-{{ block.id }}'>\n" + template_string + "\n</div>"
    return json.loads(schema_string), template_string

# This basically adds the .liquid extension to the template name since this is
# how snippets are rendered in shopify, without the extension.
class ShopifySnippetFilesystemLoader(FileSystemLoader):
    def get_source(self, _: Environment, template_name: str):
        return super().get_source(_, template_name + ".liquid")

def render_blocks(snippets_directory, blocks_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    env = Environment(loader=ShopifySnippetFilesystemLoader(snippets_directory))
    define_filters(env)

    for filename in os.listdir(blocks_directory):
        f = os.path.join(blocks_directory, filename)
        if os.path.isfile(f) and f.endswith(".liquid"):
            block_name = filename.rsplit(".", 1)[0]
            with open(f, "r") as file:
                logging.info("Rendering block %s", block_name)
                file_contents = file.read()
                schema, template_string = extract_schema(file_contents)
                template = env.from_string(template_string)
                logging.debug("Schema: %s", json.dumps(schema, indent=2))
                # We need to generate a unique id for the block div so that we can
                # target it in our tests and provide this id in the block context.
                block_params = get_block_params(block_name, schema)
                rendered_template = template.render(**block_params)
                # save this to a file of the same name in the output folder
                output_filename = block_name + ".html"
                with open(os.path.join(output_directory, output_filename), "w") as output_file:
                    output_file.write(rendered_template)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_blocks(SNIPPETS_DIRECTORY, BLOCKS_DIRECTORY, OUTPUT_DIRECTORY)