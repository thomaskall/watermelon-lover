import serial
import time


class weightSensor():
    def __init__(self, port, baudrate, timeout):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        

    @property
    def is_open(self):
        """Getter mask for the sensor property"""
        return self.ser.is_open

    def connect_serial(self):
        if self.ser.is_open:
            return
        for i in range(5):
            try:
                self.ser = serial.Serial(self.port, self.baudrate, self.timeout)
            except serial.SerialException as e:
                print(f"Error connecting to {self.port}: {e}")
                time.sleep(1)
        return self.ser.is_open

    def get_data(self) -> str:
        data = None
        try:
            data = self.ser.readline().decode('utf-8').strip()
            if data:
                print(f"Received: {data}")
        except serial.SerialException as e:
            print(f"Serial exception occurred: {e}")
            print("Attempting to reconnect...")
            ser.close()
            ser = self.connect_serial()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return data

    def close(self):
        """Closes Serial connection."""
        self.ser.close()

def main():
    sensor = weightSensor()
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
