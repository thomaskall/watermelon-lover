# Creates a instance of the Error Frame, which should show a warning and an OK button.

import customtkinter
from theme_colors import *
from PIL import Image

class frame_error(customtkinter.CTkFrame):
    def __init__(self, master, str_warning):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        #Frame consists of image and error in 1st row, and OK button on second row.
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_configure(sticky="nesw")

        error_image = customtkinter.CTkImage(light_image=Image.open("img/error.png"),
                                                  dark_image=Image.open("img/error.png"),
                                                  size=(150, 150))
        self.img_error = customtkinter.CTkLabel(self, image=error_image, text="")
        self.img_error.grid(row=0, column=0, padx=20, pady=20)

        self.lbl_message = customtkinter.CTkLabel(self, justify="center", wraplength=250,
                                                   text=str_warning)
        self.lbl_message.grid(row=0, column=1, padx=20, pady=20)

        self.btn_OK = customtkinter.CTkButton(self, text="OK", command=self.master._callback_returnFromError)
        self.btn_OK.grid(row=1, column=0, columnspan=2, padx=20, pady=20, sticky="s")

        