from serial import Serial
from serial.serialutil import SerialException
import time

class WeightSensor():
    def __init__(self, port: str, baudrate: int, timeout: int):
        self.ser: Serial
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    @property
    def is_open(self) -> bool:
        """Getter mask for the sensor property"""
        return False if (self.ser is None) else self.ser.is_open

    def connect_serial(self):
        if self.is_open:
            return
        for i in range(5):
            try:
                self.ser = Serial(self.port, self.baudrate, timeout=self.timeout)
            except SerialException as e:
                print(f"Error connecting to {self.port}: {e}")
                time.sleep(1)
        return self.ser.is_open

    def get_data(self) -> str:
        if (not self.is_open):
            print("ERROR: Serial for weight sensor is not open.")
            return
        data = None
        try:
            # Split data value from Serial string in format: "Received: 0.0000 kgs"
            data = self.ser.readline().decode('utf-8').strip().split(" ")[1]
            if data:
                print(f"Received: {data}")
                #Truncate data for negative values.
                data = int(data) if int(data) > 0 else 0
        except SerialException as e:
            print(f"Serial exception occurred: {e}")
            print("Attempting to reconnect...")
            self.ser.close()
            self.ser = self.connect_serial()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return data

    def close(self):
        """Closes Serial connection."""
        self.ser.close()

def main():
    port = '/dev/ttyUSB0' # Replace with serial port: ls /dev/tty* | grep usb
    baudrate = 9600
    timeout = 1
    sensor = WeightSensor(port, baudrate, timeout)
    sensor.connect_serial()
    try:
        while True:
            sensor.get_data()
    finally:
        if sensor.is_open:
            sensor.close()
            print("Serial port closed.")

if __name__ == "__main__":
    main()
