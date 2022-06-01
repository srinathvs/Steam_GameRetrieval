import lxml
import requests
import bs4
import random

'''
API :
________________________________________________________________________________________________________________________
get_description : Returns description as a string -

{
input : string url ; 
output : string description;
};
________________________________________________________________________________________________________________________
get_image : returns link to the image to be shown on highlight reel -

{
input : string url ; 
output : string image_url;
};
________________________________________________________________________________________________________________________
get_price : returns original price with bundles if present and discounted price(if present) -

{
input : string url ; 
output : string base_price, string discounted_price;
};
________________________________________________________________________________________________________________________
get_review_status : returns the overall review status of the game be it positive or negative

{
input : string url ; 
output : string review;
};
________________________________________________________________________________________________________________________

get_tags : returns the overall tags of the game be it positive or negative

{
input : string url ; 
output : list tags;
};
'''


# returns description from the steam store page
def get_description(src):
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    description = soup.find("meta", property="og:description")
    if description:
        description_text = description["content"]
        print(description_text)
        return description_text
    else:
        return "No description found"


# returns image from the carousel in the steam store
def get_image(src):
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    tags = soup.find_all(class_="highlight_screenshot_link")
    links = [i["href"] for i in tags]
    returned_image = "No image found"

    rand = random.randint(0, len(links)-1)
    returned_image = links[rand]

    return returned_image


# returns current prices of the game from the steam store
def get_price(src):
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    tags = soup.find_all(class_="game_purchase_price price")
    base_game_price = "Not found"
    discounted_price = "No Discount"
    if len(tags) > 0:
        base_game_price = tags[0].text.strip()
    tag2 = soup.find_all(class_="discount_final_price")
    if len(tag2) > 0:
        print(tag2[0].text)
        if base_game_price != "Free to Play":
            discounted_price = tag2[0].text.strip()

    print("No discount game price is : ", base_game_price)
    print("Discounted base price is : ", discounted_price)
    return base_game_price, discounted_price


# returns status of reviews on title
def get_review_status(src):
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    summary = soup.find(class_="game_review_summary positive")
    if summary:
        print(summary.text)
        return summary.text
    else:
        return "nothign found"


# return all tags present in the steam store
def get_all_tags():
    src = 'https://store.steampowered.com/tag/browse/#global_492'
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    tags = soup.find_all(class_="tag_browse_tag")
    tag_text = [t.text for t in tags]
    print("Total number of tags : ", len(tag_text))
    print(tag_text)
    return tag_text


# returns tags of the given url on steam
def get_tags(src):
    source = requests.get(src)
    soup = bs4.BeautifulSoup(source.content, 'html.parser')
    tags = soup.find_all(class_="glance_tags popular_tags")
    str_tagconcat = ""
    for tag in tags:
        str_tagconcat += tag.text.strip()
    str_tagconcat.rstrip()
    str_tagconcat.lstrip()
    str_tagconcat.replace("\n", "").replace("\r", "").replace('/\r/\n', "")
    tag_list = str_tagconcat.split("\t")
    tagfinal = [elem for elem in tag_list if elem != "" and elem != '\r\n' and elem != '+']
    return tagfinal


# just used for debugging, this is of no real interest
if __name__ == '__main__':
    # Free Game : https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/?snr=1_7_7_230_150_1

    # Un-discounted : https://store.steampowered.com/app/1325200/Nioh_2__The_Complete_Edition/

    # Discounted :  get_image('https://store.steampowered.com/app/1097150/Fall_Guys_Ultimate_Knockout/?snr=1_7_7_230_150_1')

    # get_price('https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/?snr=1_7_7_230_150_1')
    get_tags('https://store.steampowered.com/app/1325200/Nioh_2__The_Complete_Edition/')
    # get_all_tags()
