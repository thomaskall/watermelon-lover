UI_GREEN        = "#50C878"
PURE_GREEN      = "#00FF00"
UI_DARK_GREEN    = "#558862"
UI_RED          = "#F13D45"
PURE_RED        = "#FF0000"
UI_DARK_RED      = "#A20010"
UI_BLACK        = "#000000"
UI_GRAY         = "#6b6b6b"

# Parse hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)