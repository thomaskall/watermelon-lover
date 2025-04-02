import customtkinter
from theme_colors import *

# Creates a instance of the Results frame,

class frame_result(customtkinter.CTkFrame):

    def __init__(self, master, color_sweetness, color_quality):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")
        
        # Frame consists of 2 buttons horizontally
        self.grid_columnconfigure((0,2), weight=1)
        self.grid_rowconfigure((0,2), weight=1)
        self.grid_configure(sticky="nesw")

        # Include some comparison to change appearance of results frame depending on test results.
        self.lbl_greeting = customtkinter.CTkLabel(self, text="Test Results: ", justify="left")
        self.lbl_greeting.grid(row=0, column=0, columnspan=2)
        
        self.lbl_sweetness = customtkinter.CTkLabel(self, text="Sweetness: ", justify="left")
        self.lbl_sweetness.grid(row=1, column=0, padx=20, pady=20)

        self.canvas_sweetness = customtkinter.CTkCanvas(self, width=80, height=80, bd=0, bg="gray14", 
                                                     borderwidth=0, highlightthickness=0)
        self.canvas_sweetness.create_aa_circle(x_pos=40, y_pos=40, radius=40, 
                                            fill=color_sweetness)
        self.canvas_sweetness.grid(row=1, column=1, padx=20, pady=20)
        
        self.lbl_quality = customtkinter.CTkLabel(self, text="Quality: ", justify="left")
        self.lbl_quality.grid(row=2, column=0, padx=20, pady=20)

        self.canvas_quality = customtkinter.CTkCanvas(self, width=80, height=80, bd=0, bg="gray14", 
                                                     borderwidth=0, highlightthickness=0)
        self.canvas_quality.create_aa_circle(x_pos=40, y_pos=40, radius=40, 
                                            fill=color_quality)
        self.canvas_quality.grid(row=2, column=1, padx=20, pady=20)

        self.btn_returnHome = customtkinter.CTkButton(self, text="RETURN\nHOME", text_color=ui_darkGreen, 
                                                      fg_color=ui_green, width=150, height=140, 
                                                      command=self._callback_return)
        self.btn_returnHome.grid(row=2, rowspan=2, column=2, padx=20, pady=20, sticky="se")
    
    ###################
    ##---CALLBACKS---##
    ###################

    def _callback_return(self):
        print("clicked_returnhome")
        self.master._callback_showHome()
        