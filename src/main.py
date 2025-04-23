import customtkinter
from typing import Literal
from ui import *
from collect import *
from predict import get_spectrogram #*
from threading import Thread
from time import sleep


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

        self.isTared: bool = False

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
        else:
            print("WARNING!!! THIS FRAME IS NOT VALID")

    # Returns a hex color value that is interpolated on a range between min and max on a range.
    def _colorInterpolation(self, low_color: str, high_color: str, low: float, high: float, value: float) -> str:
        # Clamp value within bounds
        value = max(low, min(value, high))
        
        # Normalize value between 0 and 1
        t = (value - low) / (high - low)

        low_rgb = hex_to_rgb(low_color)
        high_rgb = hex_to_rgb(high_color)

        # Interpolate each color channel
        interp_rgb = tuple(
            int(low_c + (high_c - low_c) * t)
            for low_c, high_c in zip(low_rgb, high_rgb)
        )

        return rgb_to_hex(interp_rgb)

    ###################
    ##---CALLBACKS---##
    ###################

    def _show_error(self, message: str):
        print("displaying error.")
        self.prevFrame = self.shown_frame
        self.shown_frame = ErrorFrame(self, message)
        self.shown_frame.grid(row=0, column=0, sticky="nesw")
        self.shown_frame.tkraise()

    def _callback_returnFromError(self):
        temp = self.shown_frame
        self.shown_frame = self.prevFrame
        self.prevFrame = temp
        self.shown_frame.grid(row=0, column=0, sticky="nesw")
        self.shown_frame.tkraise()

    def _callback_showHome(self):
        print("showing home")
        self.show_frame(HomeFrame(self, self.isTared))

    def _callback_validate_cycle(self, cycle_type: Literal["sweep", "tap"]):
        """
        The main process that initiates a prediction cycle.
        The choice of a sweep or tap is made by the user when pressing on the corresponding button.
        """
        if not self.isTared:
            self._show_error("Please calibrate the scale before running a cycle.")
            return
        
        if cycle_type == "tap":
            self.show_frame(TapInstructionFrame(
                self, 
                title="How to Test the Watermelon",
                message="Once you click continue, tap the watermelon until time runs out.\nFor best results, either lightly slap the watermelon with an open palm or flick it.",
                on_cancel=self._callback_showHome
                ))
        else:
            self._callback_run_cycle("sweep")
        
    def _callback_run_cycle(self, cycle_type: Literal["sweep", "tap"]):
        # Collect image data and display it
        data: WatermelonData = self.data_collector.get_image_path(cycle_type, (self.width, self.height))

        # Step 2: Show countdown frame and start audio collection in background
        countdown_frame = CountdownFrame(self, cycle_type=cycle_type, duration=self.data_collector.audio_controller.audio_duration)
        self.show_frame(countdown_frame)

        # Threaded process
        def threaded_audio_collection():
            nonlocal data
            data = self.data_collector.capture_data(data)
            sleep(5)
            print(f"Weight recorded: {data.weight}")
            print(f"Audio file at: {data.wav_path}")
            print("FINISHED DATA COLLECTION")

            # TODO: Feature extraction and prediction
            data.spectrogram_path = get_spectrogram(data.wav_path)
            data.brix_prediction = 5 # predict_from_path(data.spectrogram_path, data.weight)

            # Show the result
            if data.is_complete():
                #TODO: Move this calculation to INSIDE frame_result, with watermelonData as the only input.
                color_sweetness_color: str = self._colorInterpolation(PURE_RED, PURE_GREEN, 5, 12, data.brix_prediction)
                self.show_frame(ResultFrame(self, data.brix_prediction, 5, 12, color_sweetness_color))

            self.isTared = False

        collection_thread: Thread = Thread(target=threaded_audio_collection, daemon=True)
        collection_thread.start()

        
    
    def _callback_tare(self):
        """
        Tares the scale, recalibrating it to prepare for a new cycle.
        """
        # self.data_collector.weight_sensor.tare()
        # print(self.data_collector.weight_sensor.get_data())
        self.isTared = True
        print("TARED SCALE")
        self._callback_showHome()


app = App()

app.mainloop()
