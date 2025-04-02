# Creates a instance of the Home Frame, which should show a greeting and two buttons.
# Button 1 will start a cycle.
# Button 2 will tell the scale to tare.
# Will display errors if certain conditions are not met. (not implemented yet)
import customtkinter
from theme_colors import *

class frame_home(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        #Frame consists of 2 buttons horizontally
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_configure(sticky="nesw")

        self.lbl_greeting = customtkinter.CTkLabel(self, justify="center", wraplength=430,
                                                   text="Welcome to Watermelon Lover! Please recalibrate " +
                                                   "the device before testing a watermelon.")
        self.lbl_greeting.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

        self.btn_start = customtkinter.CTkButton(self, text="START\nCYCLE", command=self._callback_start,
                                                 fg_color=ui_green, text_color=ui_darkGreen,
                                                 width=200, height=200)
        self.btn_start.grid(row=1, column=0, padx=20, pady=20, sticky="w")

        self.btn_tare = customtkinter.CTkButton(self, text="TARE\nSCALE", command=self._callback_tare,
                                                fg_color=ui_red, text_color=ui_darkRed,
                                                width=200, height=200)
        self.btn_tare.grid(row=1, column=1, padx=20, pady=20, sticky="e")

    def disable_controls(self):
        self.btn_tare.configure(state="disabled")
        self.btn_start.configure(state="disabled")

    def enable_controls(self):
        self.btn_tare.configure(state="normal")
        self.btn_start.configure(state="normal")

    ###################
    ##---CALLBACKS---##
    ###################
    
    def _callback_start(self):
        print("STARTED CYCLE")
        self.disable_controls()
        self.master._callback_runCycle()
        self.enable_controls()
        
    def _callback_tare(self):
        print("ATTEMPTED TARE OF SCALE")
        self.disable_controls()
        self.master._callback_tare()
        self.enable_controls()

        