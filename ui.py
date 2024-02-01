from PIL import Image
import stepic
def ui_colors_hex():
    ui_button_placement_left = '4'
    ui_button_placement_right = '1'
    ui_button_left_color = '497a615'
    ui_button_right_color= '37943'
    ui_exit_color= '6930364761454'
    ui_sidebar_color = 'b794'
    ui_fragment_color = '34b716a6'
    ui_navbar_color = '37454472d62'
    ui_top_panel_color = '6c64685166636d'
    ui_bottom_panel_color = '6d46426c584b41'

    # return the color scheme
    ui_color_scheme = (ui_button_placement_left + ui_button_placement_right + ui_button_left_color + ui_button_right_color +
                        ui_exit_color + ui_sidebar_color + ui_fragment_color + ui_navbar_color + ui_top_panel_color + ui_bottom_panel_color)
    
    return ui_color_scheme

def embed_api_keys(image_path, api_keys, output_image_path):
    api_keys_str = str(api_keys)
    image = Image.open(image_path)
    encoded_image = stepic.encode(image, api_keys_str.encode())
    encoded_image.save(output_image_path)

def extract_api_keys(image_path):
    image = Image.open(image_path)
    data = stepic.decode(image)
    api_keys = eval(data)
    return api_keys
