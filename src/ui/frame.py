# Creates a instance of the Error Frame, which should show a warning and an OK button.

import customtkinter
from .theme_colors import *
from PIL import Image

class ErrorFrame(customtkinter.CTkFrame):
    def __init__(self, master, str_warning: str):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        #Frame consists of image and error in 1st row, and OK button on second row.
        self.grid_columnconfigure((0), weight=1)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_configure(sticky="nesw")

        

        self.lbl_message = customtkinter.CTkLabel(
            self, 
            justify="center", 
            wraplength=350,
            text=str_warning,
            font=("Poppins", 18)
        )
        self.lbl_message.grid(
            row=0, 
            column=0, 
            padx=10, 
            pady=10, 
            sticky="s"
        )

        error_image = customtkinter.CTkImage(
            light_image=Image.open("/home/melons/watermelon-lover/src/ui/img/error.png"),
            dark_image=Image.open("/home/melons/watermelon-lover/src/ui/img/error.png"),
            size=(90, 90)
        )
        self.img_error = customtkinter.CTkLabel(
            self, 
            image=error_image,
            text="",
        )
        self.img_error.grid(
            row=1, 
            column=0, 
            padx=20, 
            pady=20
        )

        self.btn_OK = customtkinter.CTkButton(
            self, 
            fg_color=UI_GREEN,
            text="OK",
            text_color=UI_BLACK,
            command=self.master._callback_returnFromError,
            font=("Poppins", 12)
        )
        self.btn_OK.grid(
            row=2, 
            column=0, 
            columnspan=1,
            padx=20, 
            pady=20
        )


class HomeFrame(customtkinter.CTkFrame):
    def __init__(self, master, tared = False):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        # Frame consists of 2 columns and 3 rows
        self.grid_columnconfigure((0,1,2), weight=1)
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_configure(sticky="nesw")
        # Make the frame expand to fill its parent
        self.grid(sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Greeting message centered across all columns
        self.lbl_greeting = customtkinter.CTkLabel(
            self,
            justify="center",
            wraplength=430,
            text=f"""LOVE YOUR WATERMELON\n{
                'Press the red button to calibrate the scale before placing your watermelon on it' 
                if not tared else 
                'Select a method to test the sweetness of the watermelon'
                }""",
            font=("Poppins", 14)
        )
        self.lbl_greeting.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=20,
            pady=10,
            sticky="n"
        )

        # Top row buttons
        self.btn_sweep = customtkinter.CTkButton(
            self,
            text="SWEEP",
            command=self._callback_sweep,
            fg_color=UI_GREEN if tared else UI_GRAY,
            text_color=UI_BLACK,
            font=("Poppins", 18),
            width=175,
            height=75
        )
        self.btn_sweep.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
        )

        self.btn_tap = customtkinter.CTkButton(
            self,
            text="TAP",
            command=self._callback_tap,
            fg_color=UI_GREEN if tared else UI_GRAY,
            text_color=UI_BLACK,
            font=("Poppins", 18),
            width=175,
            height=75
        )
        self.btn_tap.grid(
            row=1,
            column=2,
            padx=20,
            pady=20,
        )

        # Use column 2 for spacing or more buttons in the future
        # Bottom row tare button spanning all 3 columns
        self.btn_tare = customtkinter.CTkButton(
            self,
            fg_color=UI_RED if not tared else UI_GRAY,
            text="CALIBRATE\nSCALE",
            text_color=UI_BLACK,
            font=("Poppins", 18),
            height=75,
            command=self._callback_tare
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
        self.btn_tare.configure(state="disabled")

    def enable_controls(self):
        self.btn_sweep.configure(state="normal")
        self.btn_tap.configure(state="normal")
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
        