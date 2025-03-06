import serial
import time

port = '/dev/tty.usbserial-110' # Replace with serial port: ls /dev/tty* | grep usb
baudrate = 9600
timeout = 1

def connect_serial():
    ser = None
    while ser is None:
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Connected to {port}")
            return ser
        except serial.SerialException as e:
            print(f"Error connecting to {port}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

def main():
    ser = connect_serial()
    try:
        while True:
            try:
                data = ser.readline().decode('utf-8').strip()
                if data:
                    print(f"Received: {data}")
            except serial.SerialException as e:
                print(f"Serial exception occurred: {e}")
                print("Attempting to reconnect...")
                ser.close()
                ser = connect_serial()
            except Exception as e:
                 print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        print("Keyboard interrupt exiting...")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    main()
