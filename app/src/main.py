#pip3 install customtkinter
import customtkinter
import subprocess

from typing import Literal
from ui import *
from collect import *

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


# Initialization parameters for weight sensor.
port = '/dev/ttyUSB0' # Replace with serial port: ls /dev/tty* | grep usb
baudrate = 9600
timeout = 1

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.prevFrame = customtkinter.CTkFrame
        self.data_collector = DataCollector()

        # Initialize window settings
        self.title("WatermelonLoverUI")
        self.overrideredirect(True)     # Removes window headers, use alt-F4 or close terminal to quit.
        self.geometry("480x320")        # LCD screen dimension is 480x320
        self.width: int = self.winfo_reqwidth()
        self.height: int = self.winfo_reqheight()

        # Start program showing home screen. Can be modified as needed
        self.shown_frame = HomeFrame(self)
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
        self.shown_frame = ErrorFrame(self, message)
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
        self.show_frame(HomeFrame(self))

    def _callback_runCycle(self, cycle_type: Literal["sweep", "tap"]):
        """
        The main process that initiates a prediction cycle.
        The choice of a sweep or tap is made by the user when pressing on the corresponding button.
        """
        if not self.isTared:
            self._show_error("Please calibrate the scale before running a cycle.")
            return
        
        # Collect image data and display it
        data: WatermelonData | None = self.data_collector.get_image_path(cycle_type, (self.width, self.height))
        if data.image_path is not None:
            print(f"Displaying image: {data.image_path}")
            self.show_frame(DisplayFrame(self, data.image_path))

        # Collect audio data and display the status of the collection
        data = self.data_collector.capture_data(data)
        # TODO: make audio data collection threaded...
        ### Get indicator for beginning of audio collection and length of collection
        ### Show progress bar of audio collection
        ### When audio collection is complete, show completion message
        print(f"Weight recorded: {data.weight}")
        print(f"Audio file at: {data.wav_path}")
        print("FINISHED DATA COLLECTION")

        # TODO: Feature extraction and prediction

        # Reset Tare status
        self.isTared = False

        # Show the result
        if data.is_complete():
            #TODO: Move this calculation to INSIDE frame_result, with watermelonData as the only input.
            color_sweetness = self._colorInterpolation(UI_RED, UI_GREEN, 0, 10, self.data.sweetness)
            color_quality = self._colorInterpolation(UI_RED, UI_GREEN, 0, 10, self.data.quality)
            self.show_frame(ResultFrame(self, color_sweetness, color_quality))
    
    def _callback_tare(self):
        """
        Tares the scale, recalibrating it to prepare for a new cycle.
        """
        self.data_collector.weight_sensor.tare()
        print(self.data_collector.weight_sensor.get_data())
        self.isTared = True
        print("TARED SCALE")

app = App()

app.mainloop()
