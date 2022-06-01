import os
import sys
import WebScraper
from Indexer import process_query
import PySimpleGUI as sg
import Indexer
from Indexer import data_point
from Indexer import inverted_index
from PIL import Image
import io
import requests
import WebScraper

processing = False
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
query_str = ""


def make_display():
    global processing, file_types, query_str
    sg.theme('DarkAmber')
    rlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    layout = [[sg.Text("Enter Query")],
              [sg.Input(key="Query")],
              [sg.Text("Filter Options")],
              [sg.Checkbox('Single player', key='Single player')], [sg.Checkbox('Online', key='Online')],
              [sg.Listbox(rlist, key='Show', visible=False, enable_events=True)],
              [sg.Button('Submit')],
              [sg.Multiline("No query being processed yet", key='Process', enable_events=True)]]

    window = sg.Window('Window that stays open', layout)

    while not processing:  # The Event Loop
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Submit':
            if not processing:
                print("returned values are : ", values)
                processing = True
                window['Process'].update('Processing query')
                query_string = values['Query']
                if values['Single player']:
                    query_string += " Singleplayer"
                if values['Online']:
                    query_string += " Online"
                print("Query is : ", query_string)
                query_str = query_string
                break
        elif event == 'Show':
            processing = False
            print(values['Show'])

    window.close()


def show_result_fail():
    sg.theme('DarkAmber')
    layout = [[sg.Text("The query entered did not make any sense, please try again")]]
    window1 = sg.Window('Window that shows that no result was obtained', layout)
    while True:
        event, values = window1.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
    window1.close()


def show_result_success():
    print("Success")


def Reset_Processing_State():
    global processing
    processing = False


def show_Correct_Response(url):
    index = 0
    current_process = False
    elements = [
        [sg.Image(key="-IMAGE-")],
        [sg.Multiline("Image url", key='url', enable_events=True)],
        [sg.Multiline("Reviews", key='review', enable_events=True)],
        [sg.Multiline("Price", key='price', enable_events=True)],
        [sg.Button('load Data')],
        [sg.Button('next')],
        [sg.Button('previous')],
        [sg.Multiline("Description yet to be loaded", key='desc', enable_events=True)]
    ]
    window = sg.Window("Image Viewer", elements)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "load Data" or "next" or "previous":
            if not current_process:
                current_process = True
                if event == "next":
                    if index < len(url) - 1:
                        index += 1
                    else:
                        index = 0
                if event == "previous":
                    if index == 0:
                        index = len(url) - 1
                    else:
                        index -= 1
                img_url = WebScraper.get_image(url[index])
                curr_url = url[index]
                window['url'].update(curr_url)
                desc = WebScraper.get_description(url[index])
                window['desc'].update(desc)
                review = WebScraper.get_review_status(url[index])
                window['review'].update(review)
                item_price = WebScraper.get_price(url[index])
                window['price'].update(item_price)
                image = Image.open(requests.get(img_url, stream=True).raw)
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
                current_process = False
    window.close()



if __name__ == '__main__':
    # get_image('https://store.steampowered.com/app/1097150/Fall_Guys_Ultimate_Knockout/?snr=1_7_7_230_150_1')
    response_list = []
    while not processing:
        make_display()
        print("Query string is : ", query_str)
        if query_str:
            response_list = Indexer.process_query_reduced(query_str, True)
            show_Correct_Response(response_list)
        elif query_str is None:
            show_result_fail()
        if len(response_list) == 0:
            show_result_fail()
        Reset_Processing_State()
