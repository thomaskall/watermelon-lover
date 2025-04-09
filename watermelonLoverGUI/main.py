#pip3 install customtkinter
import customtkinter
import subprocess

from frame_error    import *
from frame_home     import *
from frame_result   import *
from data           import *
from DataCollector import DataCollector
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.prevFrame = customtkinter.CTkFrame
        self.isTared = False
        self.data = data()
        self.dataCollector = DataCollector()

        # Initialize window settings
        self.title("WatermelonLoverUI")
        self.overrideredirect(True)     # Removes window headers, use alt-F4 or close terminal to quit.
        self.geometry("480x320")        # LCD screen dimension is 480x320

        # Start program showing home screen. Can be modified as needed
        self.shown_frame = frame_home(self)
        self.shown_frame.grid(row=0, column=0, sticky="nesw")
    
    # Updates the shown_frame to the provided frame. Input has to be an instance of CTkFrame
    def show_frame(self, newFrame):
        print("updating frame")
        if (isinstance(newFrame, customtkinter.CTkFrame)):
            # Updates display of window.
            self.shown_frame = newFrame
            self.shown_frame.grid(row=0, column=0, sticky="nesw")
            self.shown_frame.tkraise()

    # Returns a hex color value that is interpolated on a range between min and max on a range.
    def _colorInterpolation(self, low_color, high_color, low, high, value):
        low_color  = int(low_color[1:], 32)
        high_color = int(high_color[1:], 32)
        red     = int((((high_color & 0xff0000) - (low_color & 0xff0000)) * (value - low)) / (high - low))
        green   = int((((high_color & 0x00ff00) - (low_color & 0x00ff00)) * (value - low)) / (high - low))
        blue    = int((((high_color & 0x0000ff) - (low_color & 0x0000ff)) * (value - low)) / (high - low))
        return "#" + hex(0xffffff & ((red & 0xff0000) | (green & 0x00ff00) | (blue & 0x0000ff)))[2:]

    def _show_error(self, message):
        print("displaying error.")
        self.prevFrame = self.shown_frame
        self.shown_frame = frame_error(self, message)
        self.shown_frame.grid(row=0, column=0, sticky="nesw")
        self.shown_frame.tkraise()

    ###################
    ##---CALLBACKS---##
    ###################

    def _callback_returnFromError(self):
        temp = self.shown_frame
        self.shown_frame = self.prevFrame
        self.prevFrame = temp
        self.shown_frame.grid(row=0, column=0, sticky="nesw")
        self.shown_frame.tkraise()

    def _callback_showHome(self):
        print("showing home")
        self.data.clearData()
        self.show_frame(frame_home(self))

    def _callback_runCycle(self):
        # Check for Tared status
        if (not self.isTared):
            self._show_error("Not tared. Please remove all items from machine and " \
            "recalibrate (tare) before starting.")
            return

        # Insert test cycle here.
        #TODO: debug this on actual system, just adapted existing DataCollector code for now.
        self.dataCollector.start()
        #TODO: Figure out what data will be output by the ML model, update UI (frame_result) accordingly.
        self.data.recordData(7,5)
        print("RAN CYCLE")

        # Reset Tare status
        self.isTared = False

        # Show the result
        if (self.data.valid):
            print("1")
            color_sweetness = self._colorInterpolation(ui_red, ui_green, 0, 10, self.data.sweetness)
            color_quality = self._colorInterpolation(ui_red, ui_green, 0, 10, self.data.quality)
            self.show_frame(frame_result(self, color_sweetness, color_quality))
    
    def _callback_tare(self):
        # Placeholder for taring
        print("TARED SCALE")
        self.isTared = True

app = App()

app.mainloop()