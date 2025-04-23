# Creates a instance of the Error Frame, which should show a warning and an OK button.
from typing import Literal, Callable
import customtkinter
from .color import *
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
    def __init__(self, master, tared=False):
        super().__init__(master)
        self.master = master
        self.tared = tared
        self.configure(fg_color="transparent")

        # Grid layout to manage space
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid(sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Title
        self.lbl_title = customtkinter.CTkLabel(
            self,
            text="LOVE YOUR WATERMELON",
            font=("Poppins", 24, "bold")
        )
        self.lbl_title.pack(pady=(30, 10))

        # Message body
        message = (
            "Before placing the watermelon on the scale, press the red button to calibrate the scale."
            if not self.tared else
            "Select a method to test the sweetness of the watermelon."
        )

        self.lbl_message = customtkinter.CTkLabel(
            self,
            text=message,
            font=("Poppins", 14),
            wraplength=380,
            justify="center"
        )
        self.lbl_message.pack(pady=(10, 20), padx=20)

        # Spacer to push buttons to bottom
        self.spacer = customtkinter.CTkLabel(self, text="")
        self.spacer.pack(expand=True)

        # Button container
        self.button_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=30)

        if not self.tared:
            self.btn_tare = customtkinter.CTkButton(
                self.button_frame,
                fg_color=UI_RED,
                text="CALIBRATE SCALE",
                text_color=UI_BLACK,
                font=("Poppins", 18, "bold"),
                height=75,
                width=250,
                command=self._callback_tare
            )
            self.btn_tare.pack()
        else:
            self.btn_sweep = customtkinter.CTkButton(
                self.button_frame,
                text="SWEEP",
                command=self._callback_sweep,
                fg_color=UI_GREEN,
                text_color=UI_BLACK,
                font=("Poppins", 18, "bold"),
                width=140,
                height=75
            )
            self.btn_sweep.pack(side="left", padx=10)

            self.btn_tap = customtkinter.CTkButton(
                self.button_frame,
                text="TAP",
                command=self._callback_tap,
                fg_color=UI_GREEN,
                text_color=UI_BLACK,
                font=("Poppins", 18, "bold"),
                width=140,
                height=75
            )
            self.btn_tap.pack(side="left", padx=10)

    def disable_controls(self):
        for widget in self.button_frame.winfo_children():
            widget.configure(state="disabled")

    def enable_controls(self):
        for widget in self.button_frame.winfo_children():
            widget.configure(state="normal")

    def _callback_sweep(self):
        print("STARTED SWEEP")
        self.disable_controls()
        self.master._callback_validate_cycle("sweep")
        self.enable_controls()

    def _callback_tap(self):
        print("STARTED TAP")
        self.disable_controls()
        self.master._callback_validate_cycle("tap")
        self.enable_controls()

    def _callback_tare(self):
        print("ATTEMPTED TARE OF SCALE")
        self.disable_controls()
        self.master._callback_tare()
        self.enable_controls()


class TapInstructionFrame(customtkinter.CTkFrame):
    def __init__(self, master, title: str, message: str, on_cancel: Callable):
        super().__init__(master)

        # Title
        self.title_label = customtkinter.CTkLabel(
            self,
            text=title,
            font=("Poppins", 24, "bold"),
            anchor="center"
        )
        self.title_label.pack(pady=(20, 10), padx=20)

        # Body message
        self.body_label = customtkinter.CTkLabel(
            self,
            text=message,
            font=("Poppins", 16),
            wraplength=350,
            justify="center"
        )
        self.body_label.pack(pady=(30, 20), padx=20)

        # Spacer to push buttons to the bottom
        self.spacer = customtkinter.CTkLabel(self, text="")
        self.spacer.pack(expand=True)

        # Button frame at bottom
        self.button_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=30, anchor="s")

        self.cancel_button = customtkinter.CTkButton(
            self.button_frame,
            fg_color=UI_RED,
            text="Cancel",
            text_color=UI_BLACK,
            font=("Poppins", 18),
            command=on_cancel,
            width=150,
            height=50  # Taller button
        )
        self.cancel_button.pack(side="left", padx=10)

        self.continue_button = customtkinter.CTkButton(
            self.button_frame,
            fg_color=UI_GREEN,
            text="Continue",
            text_color=UI_BLACK,
            font=("Poppins", 18),
            command=self._callback_on_continue,
            width=150,
            height=50  # Taller button
        )
        self.continue_button.pack(side="left", padx=10)

    def _callback_on_continue(self):
        self.master._callback_run_cycle("tap")


class ResultFrame(customtkinter.CTkFrame):
    def __init__(self, master, value: float, min_val: int, max_val: int, color):
        super().__init__(master)
        self.master = master
        self.configure(border_width=0, fg_color="transparent")

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Title
        self.lbl_title = customtkinter.CTkLabel(
            self,
            text="Test Results",
            font=("Poppins", 24, "bold"),
            anchor="center"
        )
        self.lbl_title.grid(row=0, column=0, pady=(20, 10), sticky="n")

        # Normalize value to 0–10 for display
        normalized_display_value = 10 * (value - min_val) / (max_val - min_val)
        normalized_display_value = max(0, min(normalized_display_value, 10))  # clamp between 0–10

        # Sweetness label
        self.lbl_sweetness = customtkinter.CTkLabel(
            self,
            text=f"Sweetness: {normalized_display_value:.2f}/10",
            font=("Poppins", 16)
        )
        self.lbl_sweetness.grid(row=1, column=0, pady=(10, 5))

        # Progress bar (customized)

        # normalize to 0–1 for the progress bar
        progress_value = normalized_display_value / 10

        # Meter bar (progress bar)
        self.progress = customtkinter.CTkProgressBar(
            self, orientation="horizontal", width=300, height=20, progress_color=color, fg_color="gray20"
        )
        self.progress.set(progress_value)  # value between 0 and 1
        self.progress.grid(row=2, column=0, columnspan=3, padx=20, pady=(10, 0))

        # Canvas-like layout frame to position min/mid/max labels
        label_canvas = customtkinter.CTkFrame(self, width=300, height=20, fg_color="transparent")
        label_canvas.grid(row=3, column=0, columnspan=3)

        # Fixed position for min label (left)
        lbl_min = customtkinter.CTkLabel(label_canvas, text="0")
        lbl_min.place(relx=0.0, rely=0.0, anchor="nw")

        # Fixed position for max label (right)
        lbl_max = customtkinter.CTkLabel(label_canvas, text="10")
        lbl_max.place(relx=1.0, rely=0.0, anchor="ne")

        # Dynamically positioned label under the tip of the progress bar
        lbl_value = customtkinter.CTkLabel(label_canvas, text=f"{normalized_display_value:.1f}")
        lbl_value.place(relx=progress_value, rely=0.0, anchor="n")

        # Return home button
        self.btn_returnHome = customtkinter.CTkButton(
            self,
            text="RETURN\nHOME",
            text_color=UI_BLACK,
            fg_color=UI_GREEN,
            font=("Poppins", 16),
            width=180,
            height=50,
            command=self._home_button_callback
        )
        self.btn_returnHome.grid(row=4, column=0, pady=(30, 20))

    def _home_button_callback(self):
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

class CountdownFrame(customtkinter.CTkFrame):
    def __init__(self, master, cycle_type: Literal["sweep", "tap"], duration: int = 5):
        super().__init__(master)
        self.duration = duration
        self.cycle_type = cycle_type

        # Title label (large font)
        self.title_label = customtkinter.CTkLabel(
            self, 
            text="Collecting data from your watermelon..." if cycle_type == "sweep" else "Tap the watermelon!",
            font=("Poppins", 24),
            wraplength=400
        )
        self.title_label.pack(pady=(20, 20))

        # Time remaining label (smaller font)
        self.status_label = customtkinter.CTkLabel(
            self,
            text=f"Time remaining: {self.duration}s",
            font=("Poppins", 14)  # Smaller font
        )
        self.status_label.pack(pady=(0, 10))

        # Progress bar
        self.progress = customtkinter.CTkProgressBar(self, orientation="horizontal", width=300)
        self.progress.pack(pady=10)
        self.progress.set(0)

        self._update_progress()

    def _update_progress(self):
        step = 1 / self.duration
        for i in range(self.duration + 1):
            self.after(i * 1000, lambda i=i: self._update_bar(i, step))

    def _update_bar(self, second, step):
        remaining = self.duration - second
        self.status_label.configure(text=f"Time remaining: {remaining}s")
        self.progress.set(step * second)
        