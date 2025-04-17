# Creates a instance of the Error Frame, which should show a warning and an OK button.

import customtkinter
from .theme_colors import *
from PIL import Image

class ErrorFrame(customtkinter.CTkFrame):
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


class HomeFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        # Frame consists of 3 columns and 3 rows
        self.grid_columnconfigure((0,1,2), weight=1)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_configure(sticky="nesw")

        # Greeting message at the top
        self.lbl_greeting = customtkinter.CTkLabel(
            self, 
            justify="center", 
            wraplength=430,
            text="LOVER YOUR WATERMELON!\nSelect one of the options below to begin."
        )
        self.lbl_greeting.grid(
            row=0, 
            column=0, 
            columnspan=3, 
            padx=20, 
            pady=20
        )

        # Top row buttons (sweep, tap, impulse)
        self.btn_sweep = customtkinter.CTkButton(
            self, 
            text="START\nSWEEP", 
            command=self._callback_sweep,
            fg_color=UI_GREEN, 
            text_color=UI_DARK_GREEN,
            width=200,
            height=200
        )
        self.btn_sweep.grid(
            row=1, 
            column=0, 
            padx=20, 
            pady=20, 
            sticky="w"
        )

        self.btn_tap = customtkinter.CTkButton(
            self, 
            text="START\nTAP", 
            command=self._callback_tap,
            fg_color=UI_GREEN, 
            text_color=UI_DARK_GREEN,
            width=200, 
            height=200
        )
        self.btn_tap.grid(
            row=1, 
            column=1, 
            padx=20, 
            pady=20
        )

        self.btn_impulse = customtkinter.CTkButton(
            self, 
            text="START\nIMPULSE", 
            command=self._callback_impulse,
            fg_color=UI_GREEN, 
            text_color=UI_DARK_GREEN,
            width=200, 
            height=200
        )
        self.btn_impulse.grid(
            row=1, 
            column=2, 
            padx=20, 
            pady=20, 
            sticky="e"
        )

        # Bottom row tare button spanning all columns
        self.btn_tare = customtkinter.CTkButton(
            self, 
            text="TARE\nSCALE", 
            command=self._callback_tare,
            fg_color=UI_RED, 
            text_color=UI_DARK_RED,
            width=600,  # Adjusted to span the width
            height=100  # Adjusted height for bottom row
        )
        self.btn_tare.grid(
            row=2, 
            column=0, 
            columnspan=3, 
            padx=20, 
            pady=20, 
            sticky="ew"
        )

    def disable_controls(self):
        self.btn_sweep.configure(state="disabled")
        self.btn_tap.configure(state="disabled")
        self.btn_impulse.configure(state="disabled")
        self.btn_tare.configure(state="disabled")

    def enable_controls(self):
        self.btn_sweep.configure(state="normal")
        self.btn_tap.configure(state="normal")
        self.btn_impulse.configure(state="normal")
        self.btn_tare.configure(state="normal")

    def _callback_sweep(self):
        print("STARTED SWEEP")
        self.disable_controls()
        self.master._callback_runCycle("sweep")
        self.enable_controls()
    
    def _callback_tap(self):
        print("STARTED TAP")
        self.disable_controls()
        self.master._callback_runCycle("tap")
        self.enable_controls()
    
    def _callback_impulse(self):
        print("STARTED IMPULSE")
        self.disable_controls()
        self.master._callback_runCycle("impulse")
        self.enable_controls()
        
    def _callback_tare(self):
        print("ATTEMPTED TARE OF SCALE")
        self.disable_controls()
        self.master._callback_tare()
        self.enable_controls()


class ResultFrame(customtkinter.CTkFrame):
    def __init__(self, master, color_sweetness, color_quality):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")
        
        # Frame consists of 2 buttons horizontally
        self.grid_columnconfigure((0,2), weight=1)
        self.grid_rowconfigure((0,2), weight=1)
        self.grid_configure(sticky="nesw")

        # Include some comparison to change appearance of results frame depending on test results.
        self.lbl_greeting = customtkinter.CTkLabel(
            self, 
            text="Test Results: ", 
            justify="left"
        )
        self.lbl_greeting.grid(
            row=0, 
            column=0, 
            columnspan=2
        )
        self.lbl_sweetness = customtkinter.CTkLabel(
            self, text="Sweetness: ", 
            justify="left"
        )
        self.lbl_sweetness.grid(
            row=1, 
            column=0, 
            padx=20, 
            pady=20
        )
        self.canvas_sweetness = customtkinter.CTkCanvas(
            self, width=80, height=80, bd=0, bg="gray14", 
            borderwidth=0, highlightthickness=0
        )
        self.canvas_sweetness.create_aa_circle(
            x_pos=40, y_pos=40, radius=40, 
            fill=color_sweetness
        )
        self.canvas_sweetness.grid(row=1, column=1, padx=20, pady=20)
        self.lbl_quality = customtkinter.CTkLabel(self, text="Quality: ", justify="left")
        self.lbl_quality.grid(row=2, column=0, padx=20, pady=20)

        self.canvas_quality = customtkinter.CTkCanvas(
            self, width=80, height=80, bd=0, bg="gray14", 
            borderwidth=0, highlightthickness=0
        )
        self.canvas_quality.create_aa_circle(
            x_pos=40, y_pos=40, radius=40, 
            fill=color_quality
            )
        self.canvas_quality.grid(row=2, column=1, padx=20, pady=20)
        self.btn_returnHome = customtkinter.CTkButton(
            self, text="RETURN\nHOME", text_color=UI_DARK_GREEN, 
            fg_color=UI_GREEN, width=150, height=140, 
            command=self._callback_return)
        self.btn_returnHome.grid(row=2, rowspan=2, column=2, padx=20, pady=20, sticky="se")

    def _callback_return(self):
        print("clicked_returnhome")
        self.master._callback_showHome()


class DisplayFrame(customtkinter.CTkFrame):
    def __init__(self, master, image_path):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_configure(sticky="nesw")

        # Load and display the image
        display_image = customtkinter.CTkImage(
            light_image=Image.open(image_path),
            dark_image=Image.open(image_path),
            size=(400, 300)  # Adjust size as needed
        )
        self.lbl_image = customtkinter.CTkLabel(
            self, 
            image=display_image, 
            text=""
        )
        self.lbl_image.grid(
            row=0, 
            column=0, 
            padx=20, 
            pady=20, 
            sticky="nsew"
        )
        