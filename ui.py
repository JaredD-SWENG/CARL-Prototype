from PIL import Image
import stepic

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